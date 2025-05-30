#include <websocketpp/config/asio_no_tls.hpp>      // WebSocket++ config for non-TLS (plain WS)
#include <websocketpp/client.hpp>                  // WebSocket++ client implementation
#include <iostream>                                // std::cout, std::cerr
#include <string>                                  // std::string
#include <memory>                                  // std::shared_ptr, std::make_shared
#include <thread>                                  // std::thread, std::this_thread::sleep_for
#include <mutex>                                   // std::mutex, std::lock_guard, std::unique_lock
#include <condition_variable>                      // std::condition_variable
#include <chrono>                                  // std::chrono::seconds, std::chrono::milliseconds
#include <atomic>                                  // std::atomic<bool>
#include <cstdlib>                                 // std::getenv
#include <stdexcept>                               // std::exception
#include <fstream>                                 // std::ifstream
#include <sstream>                                 // std::stringstream
#include <unistd.h>                               // read, close
#include <fcntl.h>                                // open, O_RDONLY
#include <sys/stat.h>                             // mkfifo

// JSON library for message parsing and serialization
#include <nlohmann/json.hpp>

// Convenience aliases for JSON and WebSocket++ placeholders
using json = nlohmann::json;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::connection_hdl;

// Define the WebSocket++ client type using ASIO transport without TLS
typedef websocketpp::client<websocketpp::config::asio> client;
typedef client::message_ptr message_ptr;

// Named pipe path (must match Python script)
const std::string PIPE_PATH = "/tmp/gesture_pipe";

// Enumeration of gesture/command types sent from the hardware to server
enum class CommandType {
    ZOOM_IN,
    ZOOM_RESET,
    UP,
    DOWN,
    RIGHT,
    LEFT,
    THREE_GUN,
    INV_THREE_GUN,
    TWO_UP,
    ONE,
    CALL,
    LIKE,
    DISLIKE,
    ROCK,
    THREE,
    THREE2,
    TIMEOUT,
    PALM,
    TAKE_PICTURE,
    HEART,
    HEART2,
    MID_FINGER,
    THUMB_INDEX,
    HOLY,
    UNKNOWN
};

class WebSocketHardwareClient {
public:
    // Constructor: store URI and clientId, initialize state flags
    WebSocketHardwareClient(std::string uri, std::string clientId)
        : m_uri(std::move(uri))
        , m_clientId(std::move(clientId))
        , m_connected(false)
        , m_connecting(false)
        , m_reconnect_attempts(0)
        , m_max_reconnect_attempts(5)
        , m_reconnect_delay_ms(2000)
        , m_stop_requested(false)
    {
        // Reduce logging verbosity
        m_client.clear_access_channels(websocketpp::log::alevel::all);
        m_client.set_access_channels(websocketpp::log::alevel::connect);
        m_client.set_access_channels(websocketpp::log::alevel::disconnect);
        m_client.set_access_channels(websocketpp::log::alevel::app);

        // Initialize ASIO I/O service
        m_client.init_asio();

        // Register event handlers
        m_client.set_open_handler(bind(&WebSocketHardwareClient::on_open, this, _1));
        m_client.set_close_handler(bind(&WebSocketHardwareClient::on_close, this, _1));
        m_client.set_fail_handler(bind(&WebSocketHardwareClient::on_fail, this, _1));
        m_client.set_message_handler(bind(&WebSocketHardwareClient::on_message, this, _1, _2));
    }

    // Destructor: ensure graceful shutdown if still running
    ~WebSocketHardwareClient() {
        if (!m_stop_requested) {
            stop();
        }
    }

    // Attempt to establish WebSocket connection (and wait for confirmation)
    bool connect() {
        if (m_connected || m_connecting) {
            return true;  // Already in progress or connected
        }
        m_connecting = true;
        m_stop_requested = false;

        std::cout << "Attempting to connect to " << m_uri << "..." << std::endl;
        try {
            websocketpp::lib::error_code ec;
            // Create connection object
            client::connection_ptr con = m_client.get_connection(m_uri, ec);
            if (ec) {
                std::cerr << "Connect initialization error: " << ec.message() << std::endl;
                m_connecting = false;
                return false;
            }
            m_hdl = con->get_handle();
            m_client.connect(con);

            // Launch ASIO run loop on its own thread if not already running
            if (!m_client_thread.joinable()) {
                m_client_thread = std::thread([this]() {
                    try {
                        m_client.run();
                    } catch (const std::exception& e) {
                        std::cerr << "Exception in ASIO run loop: " << e.what() << std::endl;
                        std::lock_guard<std::mutex> lock(m_mutex);
                        m_connected = false;
                        m_connecting = false;
                        m_cond.notify_all();
                    }
                });
            }

            // Wait (up to 10s) for on_open to signal connection success/failure
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                if (!m_cond.wait_for(lock, std::chrono::seconds(10),
                                     [this]{ return m_connected || !m_connecting; })) {
                    std::cerr << "Connection attempt timed out." << std::endl;
                    m_connecting = false;
                    return false;
                }
            }
            m_connecting = false;
            return m_connected;

        } catch (const std::exception& e) {
            std::cerr << "Exception during connect(): " << e.what() << std::endl;
            m_connecting = false;
            return false;
        }
    }

    // Stop the WebSocket client: close connection and join thread
    void stop() {
        if (m_stop_requested) return;
        m_stop_requested = true;

        // If currently connected, send a close frame
        if (m_connected) {
            websocketpp::lib::error_code ec;
            std::cout << "Closing WebSocket connection..." << std::endl;
            try {
                if (!m_hdl.expired()) {
                    m_client.close(m_hdl, websocketpp::close::status::going_away, "Client shutdown", ec);
                    if (ec) {
                        std::cerr << "Error closing connection: " << ec.message() << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Exception while closing connection: " << e.what() << std::endl;
            }
        }
        m_connected = false;
        m_connecting = false;

        // Stop ASIO event loop
        try {
            std::cout << "Stopping WebSocket ASIO service..." << std::endl;
            m_client.stop();
        } catch (const std::exception& e) {
            std::cerr << "Exception during client stop(): " << e.what() << std::endl;
        }

        // Wait for the ASIO thread to finish
        if (m_client_thread.joinable()) {
            std::cout << "Waiting for ASIO thread to join..." << std::endl;
            m_client_thread.join();
            std::cout << "WebSocket client ASIO thread joined." << std::endl;
        }
    }

    // Send a gesture command to the server encoded as JSON
    bool sendCommand(CommandType command_type, const json& position_data = json()) {
        if (!m_connected) return false;

        std::string command_str = commandTypeToString(command_type);
        if (command_str == "unknown") return false;

        // Build JSON message
        json message = {
            {"command", command_str},
        };

        // Add position data if provided (for tracking commands)
        if (!position_data.empty()) {
            message["position"] = position_data;
        }

        websocketpp::lib::error_code ec;
        try {
            if (!m_hdl.expired()) {
                m_client.send(m_hdl, message.dump(), websocketpp::frame::opcode::text, ec);
            } else {
                return false;
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception during sendCommand: " << e.what() << std::endl;
            return false;
        }

        if (ec) {
            std::cerr << "Error sending command: " << ec.message() << std::endl;
            return false;
        }
        return true;
    }

    // Check current connection state
    bool isConnected() const {
        return m_connected;
    }

    // Convert string command to CommandType enum
    CommandType stringToCommandType(const std::string& command) {
        if (command == "zoom_in")           return CommandType::ZOOM_IN;
        if (command == "zoom_reset")        return CommandType::ZOOM_RESET;
        if (command == "up")                return CommandType::UP;
        if (command == "down")              return CommandType::DOWN;
        if (command == "right")             return CommandType::RIGHT;
        if (command == "left")              return CommandType::LEFT;
        if (command == "three_gun")         return CommandType::THREE_GUN;
        if (command == "inv_three_gun")     return CommandType::INV_THREE_GUN;
        if (command == "two_up")            return CommandType::TWO_UP;
        if (command == "one")               return CommandType::ONE;
        if (command == "call")              return CommandType::CALL;
        if (command == "like")              return CommandType::LIKE;
        if (command == "dislike")           return CommandType::DISLIKE;
        if (command == "rock")              return CommandType::ROCK;
        if (command == "three")             return CommandType::THREE;
        if (command == "three2")            return CommandType::THREE2;
        if (command == "timeout")           return CommandType::TIMEOUT;
        if (command == "palm")              return CommandType::PALM;
        if (command == "take_picture")      return CommandType::TAKE_PICTURE;
        if (command == "heart")             return CommandType::HEART;
        if (command == "heart2")            return CommandType::HEART2;
        if (command == "mid_finger")        return CommandType::MID_FINGER;
        if (command == "thumb_index")       return CommandType::THUMB_INDEX;
        if (command == "holy")              return CommandType::HOLY;
        return CommandType::UNKNOWN;
    }


    // Convert CommandType enum to the corresponding string
    std::string commandTypeToString(CommandType command) {
        switch (command) {
            case CommandType::ZOOM_IN:         return "zoom_in";
            case CommandType::ZOOM_RESET:      return "zoom_reset";
            case CommandType::UP:              return "up";
            case CommandType::DOWN:            return "down";
            case CommandType::RIGHT:           return "right";
            case CommandType::LEFT:            return "left";
            case CommandType::THREE_GUN:       return "three_gun";
            case CommandType::INV_THREE_GUN:   return "inv_three_gun";
            case CommandType::TWO_UP:          return "two_up";
            case CommandType::ONE:             return "one";
            case CommandType::CALL:            return "call";
            case CommandType::LIKE:            return "like";
            case CommandType::DISLIKE:         return "dislike";
            case CommandType::ROCK:            return "rock";
            case CommandType::THREE:           return "three";
            case CommandType::THREE2:          return "three2";
            case CommandType::TIMEOUT:         return "timeout";
            case CommandType::PALM:            return "palm";
            case CommandType::TAKE_PICTURE:    return "take_picture";
            case CommandType::HEART:           return "heart";
            case CommandType::HEART2:          return "heart2";
            case CommandType::MID_FINGER:      return "mid_finger";
            case CommandType::THUMB_INDEX:     return "thumb_index";
            case CommandType::HOLY:            return "holy";
            case CommandType::UNKNOWN:         return "unknown";
        }
        return "unknown";
    }


private:
    // Called when the WebSocket connection is successfully opened
    void on_open(connection_hdl hdl) {
        std::cout << "Connection established." << std::endl;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_connected = true;
            m_connecting = false;
            m_reconnect_attempts = 0;
        }
        m_cond.notify_all();

        // Immediately send registration JSON to identify as hardware client
        json registration_msg = {
            {"register", "hardware"},
            {"id", m_clientId}
        };
        websocketpp::lib::error_code ec;
        try {
            if (!hdl.expired()) {
                m_client.send(hdl, registration_msg.dump(), websocketpp::frame::opcode::text, ec);
                if (ec) {
                    std::cerr << "Failed to send registration: " << ec.message() << std::endl;
                } else {
                    std::cout << "Sent registration request." << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception sending registration: " << e.what() << std::endl;
        }
    }

    // Called when the WebSocket handshake or connection fails
    void on_fail(connection_hdl hdl) {
        std::string error_msg = "N/A";
        auto con = m_client.get_con_from_hdl(hdl);
        if (con) {
            error_msg = con->get_ec().message();
        }
        std::cerr << "Connection attempt failed: " << error_msg << std::endl;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_connected = false;
            m_connecting = false;
        }
        m_cond.notify_all();
        schedule_reconnect();  // Try again later
    }

    // Called when an established WebSocket connection closes
    void on_close(connection_hdl hdl) {
        std::string reason = "N/A";
        auto con = m_client.get_con_from_hdl(hdl);
        if (con) {
            reason = con->get_remote_close_reason();
        }
        std::cout << "Connection closed. Reason: " << (reason.empty() ? "(unknown)" : reason) << std::endl;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_connected = false;
            m_connecting = false;
        }
        m_cond.notify_all();
        if (!m_stop_requested) {
            schedule_reconnect();  // Attempt to reconnect if not shutting down
        }
    }

    // Called when a message arrives from the server
    void on_message(connection_hdl hdl, message_ptr msg) {
        const std::string& payload = msg->get_payload();
        std::cout << "Received message from server: " << payload << std::endl;
        try {
            json data = json::parse(payload);
            if (data.contains("type")) {
                std::string type = data["type"];
                if (type == "registration_success") {
                    std::cout << "Registered successfully as ID: "
                              << data.value("client_id", "[N/A]") << std::endl;
                } else if (type == "error") {
                    std::cerr << "Server Error: "
                              << data.value("message", "(No details)") << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error processing server message: " << e.what() << std::endl;
        }
    }

    // Schedule a reconnect attempt with exponential backoff
    void schedule_reconnect() {
        if (m_stop_requested || m_connected || m_connecting) return;
        m_reconnect_attempts++;
        if (m_reconnect_attempts > m_max_reconnect_attempts) {
            std::cerr << "Max reconnect attempts reached. Giving up." << std::endl;
            return;
        }
        long long delay = m_reconnect_delay_ms * (1 << std::min(m_reconnect_attempts - 1, 4));
        std::cout << "Reconnect attempt " << m_reconnect_attempts
                  << "/" << m_max_reconnect_attempts
                  << " in " << delay << "ms..." << std::endl;
        std::thread([this, delay]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
            if (!m_stop_requested && !m_connected && !m_connecting) {
                connect();
            }
        }).detach();
    }

    // Member variables for the WebSocket++ client, state flags, and synchronization
    client                     m_client;                 // WebSocket++ client object
    connection_hdl             m_hdl;                    // Handle to the active connection
    std::thread                m_client_thread;          // Thread running the ASIO loop
    std::string                m_uri;                    // Server URI (ws://...)
    std::string                m_clientId;               // Unique hardware client ID
    std::atomic<bool>          m_connected;              // True if handshake completed
    std::atomic<bool>          m_connecting;             // True while attempting to connect
    std::atomic<bool>          m_stop_requested;         // True when shutting down
    int                        m_reconnect_attempts;     // How many times we've retried
    const int                  m_max_reconnect_attempts; // Cap for retries
    const int                  m_reconnect_delay_ms;     // Base delay between retries
    std::mutex                 m_mutex;                  // Synchronizes state flags
    std::condition_variable    m_cond;                   // Signals connect/open events
};

// Gesture Control System that reads from named pipe and sends to WebSocket
class GestureControlSystem {
public:
    GestureControlSystem(const std::string& serverUri, const std::string& clientId)
        : m_webSocketClient(serverUri, clientId), m_isRunning(false), m_pipefd(-1)
    {}

    ~GestureControlSystem() {
        stop();
    }

    // Initialize hardware resources and connect to server
    bool initialize() {
        std::cout << "Initializing Gesture Control System..." << std::endl;
        
        // Wait for the Python script to create the pipe
        std::cout << "Waiting for Python gesture recognition system..." << std::endl;
        int attempts = 0;
        while (attempts < 30) {  // Wait up to 30 seconds
            if (access(PIPE_PATH.c_str(), F_OK) == 0) {
                std::cout << "Found named pipe: " << PIPE_PATH << std::endl;
                break;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
            attempts++;
        }
        
        if (attempts >= 30) {
            std::cerr << "Timeout waiting for Python script to create pipe: " << PIPE_PATH << std::endl;
            return false;
        }

        // Open the named pipe for reading
        m_pipefd = open(PIPE_PATH.c_str(), O_RDONLY);
        if (m_pipefd == -1) {
            perror("Failed to open named pipe");
            return false;
        }
        std::cout << "Opened named pipe for reading." << std::endl;

        std::cout << "Attempting WebSocket connection..." << std::endl;
        return m_webSocketClient.connect();
    }

    // Start the processing thread to read from pipe and send to WebSocket
    void start() {
        if (!m_webSocketClient.isConnected()) {
            std::cerr << "Cannot start: WebSocket not connected." << std::endl;
            return;
        }
        if (m_isRunning) return;
        m_isRunning = true;
        m_processingThread = std::thread(&GestureControlSystem::processingLoop, this);
        std::cout << "Gesture processing loop started." << std::endl;
    }

    // Stop processing and shut down the WebSocket client
    void stop() {
        if (!m_isRunning) return;
        m_isRunning = false;
        
        if (m_processingThread.joinable()) {
            m_processingThread.join();
        }
        
        if (m_pipefd != -1) {
            close(m_pipefd);
            m_pipefd = -1;
        }
        
        m_webSocketClient.stop();
        std::cout << "Gesture Control System stopped." << std::endl;
    }

private:
    // Main loop: read from named pipe and forward to WebSocket
    void processingLoop() {
        std::cout << "Starting to listen for gesture commands from Python..." << std::endl;
        
        char buffer[1024];
        std::string line_buffer;
        
        while (m_isRunning) {
            // Read data from pipe
            ssize_t bytes_read = read(m_pipefd, buffer, sizeof(buffer) - 1);
            
            if (bytes_read > 0) {
                buffer[bytes_read] = '\0';
                std::cout << "Raw data from pipe: " << buffer << std::endl; // DEBUG LINE
                line_buffer += buffer;
                
                // Process complete lines (JSON messages end with newline)
                size_t pos;
                while ((pos = line_buffer.find('\n')) != std::string::npos) {
                    std::string json_line = line_buffer.substr(0, pos);
                    line_buffer.erase(0, pos + 1);
                    
                    if (!json_line.empty()) {
                        std::cout << "Processing line: " << json_line << std::endl; // DEBUG LINE
                        processGestureMessage(json_line);
                    }
                }
            } else if (bytes_read == 0) {
                // EOF - Python script closed the pipe
                std::cout << "Python script closed the pipe. Waiting for reconnection..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                
                // Try to reopen the pipe
                close(m_pipefd);
                m_pipefd = open(PIPE_PATH.c_str(), O_RDONLY);
                if (m_pipefd == -1) {
                    std::cerr << "Failed to reopen pipe. Exiting..." << std::endl;
                    break;
                }
            } else {
                // Error reading from pipe
                if (m_isRunning) {
                    perror("Error reading from pipe");
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
        }
        std::cout << "Exiting processing loop." << std::endl;
    }

    // Process a JSON message received from Python
    void processGestureMessage(const std::string& json_str) {
        try {
            // Try to parse as JSON first
            json data;
            try {
                data = json::parse(json_str);
            } catch (const json::parse_error& e) {
                // If not valid JSON, try to interpret as a simple command string
                // This is to handle the case where Python just sends "two_up" instead of proper JSON
                std::string command = json_str;
                // Trim whitespace
                command.erase(0, command.find_first_not_of(" \t\r\n"));
                command.erase(command.find_last_not_of(" \t\r\n") + 1);
                
                // Create a JSON object manually
                data = {
                    {"type", "gesture"},
                    {"command", command}
                };
                
                std::cout << "Converted plain text to JSON: " << data.dump() << std::endl;
            }
            
            // Now process the JSON message
            if (data.contains("type") && data["type"] == "gesture") {
                std::string command = data.value("command", "");
                CommandType cmd_type = m_webSocketClient.stringToCommandType(command);
                
                if (cmd_type != CommandType::UNKNOWN) {
                    std::cout << "Received gesture: " << command << std::endl;
                    
                    // Handle position data for tracking commands
                    json position_data;
                    if (data.contains("position")) {
                        position_data = data["position"];
                    }
                    
                    // Send command to WebSocket server
                    if (m_webSocketClient.isConnected()) {
                        bool sent = m_webSocketClient.sendCommand(cmd_type, position_data);
                        if (!sent) {
                            std::cerr << "Failed to send command: " << command << std::endl;
                        } else {
                            std::cout << "Successfully sent command to server: " << command << std::endl;
                        }
                    } else {
                        std::cout << "WebSocket not connected. Skipping command: " << command << std::endl;
                    }
                } else {
                    std::cout << "Unknown gesture command: " << command << std::endl;
                }
            } else if (!data.contains("type")) {
                // If there's no type field but it parsed as JSON, try to extract a command field
                if (data.contains("command")) {
                    std::string command = data["command"];
                    CommandType cmd_type = m_webSocketClient.stringToCommandType(command);
                    
                    if (cmd_type != CommandType::UNKNOWN) {
                        std::cout << "Received direct command JSON: " << command << std::endl;
                        
                        // Handle position data for tracking commands
                        json position_data;
                        if (data.contains("position")) {
                            position_data = data["position"];
                        }
                        
                        // Send command to WebSocket server
                        if (m_webSocketClient.isConnected()) {
                            bool sent = m_webSocketClient.sendCommand(cmd_type, position_data);
                            if (!sent) {
                                std::cerr << "Failed to send command: " << command << std::endl;
                            } else {
                                std::cout << "Successfully sent command to server: " << command << std::endl;
                            }
                        } else {
                            std::cout << "WebSocket not connected. Skipping command: " << command << std::endl;
                        }
                    } else {
                        std::cout << "Unknown gesture command: " << command << std::endl;
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error processing message: " << e.what() << std::endl;
            std::cerr << "Message was: " << json_str << std::endl;
        }
    }

    WebSocketHardwareClient    m_webSocketClient;  // Underlying WS client
    std::thread                m_processingThread; // Thread for the loop
    std::atomic<bool>          m_isRunning;        // Loop control flag
    int                        m_pipefd;           // File descriptor for named pipe
};

int main(int argc, char* argv[]) {
    // Default server URI and hardware client ID
    std::string serverUri = "ws://localhost:8080";
    std::string clientId  = "hardware-pi-01";

    // Override defaults via command-line arguments
    if (argc > 1) serverUri = argv[1];
    if (argc > 2) clientId  = argv[2];

    std::cout << "--- AirClass Hardware Client ---" << std::endl;
    std::cout << "Server URI: " << serverUri << std::endl;
    std::cout << "Client ID : " << clientId << std::endl;
    std::cout << "Named Pipe: " << PIPE_PATH << std::endl;

    // Instantiate and initialize the gesture system
    GestureControlSystem gestureSystem(serverUri, clientId);
    if (!gestureSystem.initialize()) {
        std::cerr << "FATAL: Could not initialize hardware client. Exiting." << std::endl;
        return 1;
    }

    // Begin gesture processing
    gestureSystem.start();
    

    
    // Wait for user input to terminate
    std::cout << "Hardware client running. Press Enter to exit." << std::endl;
    std::cin.get();

    std::cout << "Shutdown requested..." << std::endl;
    gestureSystem.stop();
    std::cout << "Hardware client finished." << std::endl;
    return 0;
}
#include <websocketpp/config/asio_no_tls.hpp>      // WebSocket++ config for non-TLS (plain WS)
#include <websocketpp/client.hpp>                  // WebSocket++ client implementation
#include <iostream>                                // std::cout, std::cerr
#include <string>                                  // std::string
#include <memory>                                  // std::shared_ptr, std::make_shared
#include <thread>                                  // std::thread, std::this_thread::sleep_for
#include <mutex>                                   // std::mutex, std::lock_guard, std::unique_lock
#include <condition_variable>                      // std::condition_variable
#include <chrono>                                  // std::chrono::seconds, std::chrono::milliseconds
#include <atomic>                                  // std::atomic<bool>
#include <cstdlib>                                 // std::getenv
#include <stdexcept>                               // std::exception
#include <fstream>                                 // std::ifstream
#include <sstream>                                 // std::stringstream
#include <unistd.h>                               // read, close
#include <fcntl.h>                                // open, O_RDONLY
#include <sys/stat.h>                             // mkfifo

// JSON library for message parsing and serialization
#include <nlohmann/json.hpp>

// Convenience aliases for JSON and WebSocket++ placeholders
using json = nlohmann::json;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::connection_hdl;

// Define the WebSocket++ client type using ASIO transport without TLS
typedef websocketpp::client<websocketpp::config::asio> client;
typedef client::message_ptr message_ptr;

// Named pipe path (must match Python script)
const std::string PIPE_PATH = "/tmp/gesture_pipe";

// Enumeration of gesture/command types sent from the hardware to server
enum class CommandType {
    ZOOM_IN,
    ZOOM_RESET,
    UP,
    DOWN,
    RIGHT,
    LEFT,
    THREE_GUN,
    INV_THREE_GUN,
    TWO_UP,
    ONE,
    CALL,
    LIKE,
    DISLIKE,
    ROCK,
    THREE,
    THREE2,
    TIMEOUT,
    PALM,
    TAKE_PICTURE,
    HEART,
    HEART2,
    MID_FINGER,
    THUMB_INDEX,
    HOLY,
    UNKNOWN
};

class WebSocketHardwareClient {
public:
    // Constructor: store URI and clientId, initialize state flags
    WebSocketHardwareClient(std::string uri, std::string clientId)
        : m_uri(std::move(uri))
        , m_clientId(std::move(clientId))
        , m_connected(false)
        , m_connecting(false)
        , m_reconnect_attempts(0)
        , m_max_reconnect_attempts(5)
        , m_reconnect_delay_ms(2000)
        , m_stop_requested(false)
    {
        // Reduce logging verbosity
        m_client.clear_access_channels(websocketpp::log::alevel::all);
        m_client.set_access_channels(websocketpp::log::alevel::connect);
        m_client.set_access_channels(websocketpp::log::alevel::disconnect);
        m_client.set_access_channels(websocketpp::log::alevel::app);

        // Initialize ASIO I/O service
        m_client.init_asio();

        // Register event handlers
        m_client.set_open_handler(bind(&WebSocketHardwareClient::on_open, this, _1));
        m_client.set_close_handler(bind(&WebSocketHardwareClient::on_close, this, _1));
        m_client.set_fail_handler(bind(&WebSocketHardwareClient::on_fail, this, _1));
        m_client.set_message_handler(bind(&WebSocketHardwareClient::on_message, this, _1, _2));
    }

    // Destructor: ensure graceful shutdown if still running
    ~WebSocketHardwareClient() {
        if (!m_stop_requested) {
            stop();
        }
    }

    // Attempt to establish WebSocket connection (and wait for confirmation)
    bool connect() {
        if (m_connected || m_connecting) {
            return true;  // Already in progress or connected
        }
        m_connecting = true;
        m_stop_requested = false;

        std::cout << "Attempting to connect to " << m_uri << "..." << std::endl;
        try {
            websocketpp::lib::error_code ec;
            // Create connection object
            client::connection_ptr con = m_client.get_connection(m_uri, ec);
            if (ec) {
                std::cerr << "Connect initialization error: " << ec.message() << std::endl;
                m_connecting = false;
                return false;
            }
            m_hdl = con->get_handle();
            m_client.connect(con);

            // Launch ASIO run loop on its own thread if not already running
            if (!m_client_thread.joinable()) {
                m_client_thread = std::thread([this]() {
                    try {
                        m_client.run();
                    } catch (const std::exception& e) {
                        std::cerr << "Exception in ASIO run loop: " << e.what() << std::endl;
                        std::lock_guard<std::mutex> lock(m_mutex);
                        m_connected = false;
                        m_connecting = false;
                        m_cond.notify_all();
                    }
                });
            }

            // Wait (up to 10s) for on_open to signal connection success/failure
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                if (!m_cond.wait_for(lock, std::chrono::seconds(10),
                                     [this]{ return m_connected || !m_connecting; })) {
                    std::cerr << "Connection attempt timed out." << std::endl;
                    m_connecting = false;
                    return false;
                }
            }
            m_connecting = false;
            return m_connected;

        } catch (const std::exception& e) {
            std::cerr << "Exception during connect(): " << e.what() << std::endl;
            m_connecting = false;
            return false;
        }
    }

    // Stop the WebSocket client: close connection and join thread
    void stop() {
        if (m_stop_requested) return;
        m_stop_requested = true;

        // If currently connected, send a close frame
        if (m_connected) {
            websocketpp::lib::error_code ec;
            std::cout << "Closing WebSocket connection..." << std::endl;
            try {
                if (!m_hdl.expired()) {
                    m_client.close(m_hdl, websocketpp::close::status::going_away, "Client shutdown", ec);
                    if (ec) {
                        std::cerr << "Error closing connection: " << ec.message() << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Exception while closing connection: " << e.what() << std::endl;
            }
        }
        m_connected = false;
        m_connecting = false;

        // Stop ASIO event loop
        try {
            std::cout << "Stopping WebSocket ASIO service..." << std::endl;
            m_client.stop();
        } catch (const std::exception& e) {
            std::cerr << "Exception during client stop(): " << e.what() << std::endl;
        }

        // Wait for the ASIO thread to finish
        if (m_client_thread.joinable()) {
            std::cout << "Waiting for ASIO thread to join..." << std::endl;
            m_client_thread.join();
            std::cout << "WebSocket client ASIO thread joined." << std::endl;
        }
    }

    // Send a gesture command to the server encoded as JSON
    bool sendCommand(CommandType command_type, const json& position_data = json()) {
        if (!m_connected) return false;

        std::string command_str = commandTypeToString(command_type);
        if (command_str == "unknown") return false;

        // Build JSON message
        json message = {
            {"command", command_str},
        };

        // Add position data if provided (for tracking commands)
        if (!position_data.empty()) {
            message["position"] = position_data;
        }

        websocketpp::lib::error_code ec;
        try {
            if (!m_hdl.expired()) {
                m_client.send(m_hdl, message.dump(), websocketpp::frame::opcode::text, ec);
            } else {
                return false;
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception during sendCommand: " << e.what() << std::endl;
            return false;
        }

        if (ec) {
            std::cerr << "Error sending command: " << ec.message() << std::endl;
            return false;
        }
        return true;
    }

    // Check current connection state
    bool isConnected() const {
        return m_connected;
    }

    // Convert string command to CommandType enum
    CommandType stringToCommandType(const std::string& command) {
        if (command == "zoom_in")           return CommandType::ZOOM_IN;
        if (command == "zoom_reset")        return CommandType::ZOOM_RESET;
        if (command == "up")                return CommandType::UP;
        if (command == "down")              return CommandType::DOWN;
        if (command == "right")             return CommandType::RIGHT;
        if (command == "left")              return CommandType::LEFT;
        if (command == "three_gun")         return CommandType::THREE_GUN;
        if (command == "inv_three_gun")     return CommandType::INV_THREE_GUN;
        if (command == "two_up")            return CommandType::TWO_UP;
        if (command == "one")               return CommandType::ONE;
        if (command == "call")              return CommandType::CALL;
        if (command == "like")              return CommandType::LIKE;
        if (command == "dislike")           return CommandType::DISLIKE;
        if (command == "rock")              return CommandType::ROCK;
        if (command == "three")             return CommandType::THREE;
        if (command == "three2")            return CommandType::THREE2;
        if (command == "timeout")           return CommandType::TIMEOUT;
        if (command == "palm")              return CommandType::PALM;
        if (command == "take_picture")      return CommandType::TAKE_PICTURE;
        if (command == "heart")             return CommandType::HEART;
        if (command == "heart2")            return CommandType::HEART2;
        if (command == "mid_finger")        return CommandType::MID_FINGER;
        if (command == "thumb_index")       return CommandType::THUMB_INDEX;
        if (command == "holy")              return CommandType::HOLY;
        return CommandType::UNKNOWN;
    }


    // Convert CommandType enum to the corresponding string
    std::string commandTypeToString(CommandType command) {
        switch (command) {
            case CommandType::ZOOM_IN:         return "zoom_in";
            case CommandType::ZOOM_RESET:      return "zoom_reset";
            case CommandType::UP:              return "up";
            case CommandType::DOWN:            return "down";
            case CommandType::RIGHT:           return "right";
            case CommandType::LEFT:            return "left";
            case CommandType::THREE_GUN:       return "three_gun";
            case CommandType::INV_THREE_GUN:   return "inv_three_gun";
            case CommandType::TWO_UP:          return "two_up";
            case CommandType::ONE:             return "one";
            case CommandType::CALL:            return "call";
            case CommandType::LIKE:            return "like";
            case CommandType::DISLIKE:         return "dislike";
            case CommandType::ROCK:            return "rock";
            case CommandType::THREE:           return "three";
            case CommandType::THREE2:          return "three2";
            case CommandType::TIMEOUT:         return "timeout";
            case CommandType::PALM:            return "palm";
            case CommandType::TAKE_PICTURE:    return "take_picture";
            case CommandType::HEART:           return "heart";
            case CommandType::HEART2:          return "heart2";
            case CommandType::MID_FINGER:      return "mid_finger";
            case CommandType::THUMB_INDEX:     return "thumb_index";
            case CommandType::HOLY:            return "holy";
            case CommandType::UNKNOWN:         return "unknown";
        }
        return "unknown";
    }


private:
    // Called when the WebSocket connection is successfully opened
    void on_open(connection_hdl hdl) {
        std::cout << "Connection established." << std::endl;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_connected = true;
            m_connecting = false;
            m_reconnect_attempts = 0;
        }
        m_cond.notify_all();

        // Immediately send registration JSON to identify as hardware client
        json registration_msg = {
            {"register", "hardware"},
            {"id", m_clientId}
        };
        websocketpp::lib::error_code ec;
        try {
            if (!hdl.expired()) {
                m_client.send(hdl, registration_msg.dump(), websocketpp::frame::opcode::text, ec);
                if (ec) {
                    std::cerr << "Failed to send registration: " << ec.message() << std::endl;
                } else {
                    std::cout << "Sent registration request." << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception sending registration: " << e.what() << std::endl;
        }
    }

    // Called when the WebSocket handshake or connection fails
    void on_fail(connection_hdl hdl) {
        std::string error_msg = "N/A";
        auto con = m_client.get_con_from_hdl(hdl);
        if (con) {
            error_msg = con->get_ec().message();
        }
        std::cerr << "Connection attempt failed: " << error_msg << std::endl;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_connected = false;
            m_connecting = false;
        }
        m_cond.notify_all();
        schedule_reconnect();  // Try again later
    }

    // Called when an established WebSocket connection closes
    void on_close(connection_hdl hdl) {
        std::string reason = "N/A";
        auto con = m_client.get_con_from_hdl(hdl);
        if (con) {
            reason = con->get_remote_close_reason();
        }
        std::cout << "Connection closed. Reason: " << (reason.empty() ? "(unknown)" : reason) << std::endl;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_connected = false;
            m_connecting = false;
        }
        m_cond.notify_all();
        if (!m_stop_requested) {
            schedule_reconnect();  // Attempt to reconnect if not shutting down
        }
    }

    // Called when a message arrives from the server
    void on_message(connection_hdl hdl, message_ptr msg) {
        const std::string& payload = msg->get_payload();
        std::cout << "Received message from server: " << payload << std::endl;
        try {
            json data = json::parse(payload);
            if (data.contains("type")) {
                std::string type = data["type"];
                if (type == "registration_success") {
                    std::cout << "Registered successfully as ID: "
                              << data.value("client_id", "[N/A]") << std::endl;
                } else if (type == "error") {
                    std::cerr << "Server Error: "
                              << data.value("message", "(No details)") << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error processing server message: " << e.what() << std::endl;
        }
    }

    // Schedule a reconnect attempt with exponential backoff
    void schedule_reconnect() {
        if (m_stop_requested || m_connected || m_connecting) return;
        m_reconnect_attempts++;
        if (m_reconnect_attempts > m_max_reconnect_attempts) {
            std::cerr << "Max reconnect attempts reached. Giving up." << std::endl;
            return;
        }
        long long delay = m_reconnect_delay_ms * (1 << std::min(m_reconnect_attempts - 1, 4));
        std::cout << "Reconnect attempt " << m_reconnect_attempts
                  << "/" << m_max_reconnect_attempts
                  << " in " << delay << "ms..." << std::endl;
        std::thread([this, delay]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
            if (!m_stop_requested && !m_connected && !m_connecting) {
                connect();
            }
        }).detach();
    }

    // Member variables for the WebSocket++ client, state flags, and synchronization
    client                     m_client;                 // WebSocket++ client object
    connection_hdl             m_hdl;                    // Handle to the active connection
    std::thread                m_client_thread;          // Thread running the ASIO loop
    std::string                m_uri;                    // Server URI (ws://...)
    std::string                m_clientId;               // Unique hardware client ID
    std::atomic<bool>          m_connected;              // True if handshake completed
    std::atomic<bool>          m_connecting;             // True while attempting to connect
    std::atomic<bool>          m_stop_requested;         // True when shutting down
    int                        m_reconnect_attempts;     // How many times we've retried
    const int                  m_max_reconnect_attempts; // Cap for retries
    const int                  m_reconnect_delay_ms;     // Base delay between retries
    std::mutex                 m_mutex;                  // Synchronizes state flags
    std::condition_variable    m_cond;                   // Signals connect/open events
};

// Gesture Control System that reads from named pipe and sends to WebSocket
class GestureControlSystem {
public:
    GestureControlSystem(const std::string& serverUri, const std::string& clientId)
        : m_webSocketClient(serverUri, clientId), m_isRunning(false), m_pipefd(-1)
    {}

    ~GestureControlSystem() {
        stop();
    }

    // Initialize hardware resources and connect to server
    bool initialize() {
        std::cout << "Initializing Gesture Control System..." << std::endl;
        
        // Wait for the Python script to create the pipe
        std::cout << "Waiting for Python gesture recognition system..." << std::endl;
        int attempts = 0;
        while (attempts < 30) {  // Wait up to 30 seconds
            if (access(PIPE_PATH.c_str(), F_OK) == 0) {
                std::cout << "Found named pipe: " << PIPE_PATH << std::endl;
                break;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
            attempts++;
        }
        
        if (attempts >= 30) {
            std::cerr << "Timeout waiting for Python script to create pipe: " << PIPE_PATH << std::endl;
            return false;
        }

        // Open the named pipe for reading
        m_pipefd = open(PIPE_PATH.c_str(), O_RDONLY);
        if (m_pipefd == -1) {
            perror("Failed to open named pipe");
            return false;
        }
        std::cout << "Opened named pipe for reading." << std::endl;

        std::cout << "Attempting WebSocket connection..." << std::endl;
        return m_webSocketClient.connect();
    }

    // Start the processing thread to read from pipe and send to WebSocket
    void start() {
        if (!m_webSocketClient.isConnected()) {
            std::cerr << "Cannot start: WebSocket not connected." << std::endl;
            return;
        }
        if (m_isRunning) return;
        m_isRunning = true;
        m_processingThread = std::thread(&GestureControlSystem::processingLoop, this);
        std::cout << "Gesture processing loop started." << std::endl;
    }

    // Stop processing and shut down the WebSocket client
    void stop() {
        if (!m_isRunning) return;
        m_isRunning = false;
        
        if (m_processingThread.joinable()) {
            m_processingThread.join();
        }
        
        if (m_pipefd != -1) {
            close(m_pipefd);
            m_pipefd = -1;
        }
        
        m_webSocketClient.stop();
        std::cout << "Gesture Control System stopped." << std::endl;
    }

private:
    // Main loop: read from named pipe and forward to WebSocket
    void processingLoop() {
        std::cout << "Starting to listen for gesture commands from Python..." << std::endl;
        
        char buffer[1024];
        std::string line_buffer;
        
        while (m_isRunning) {
            // Read data from pipe
            ssize_t bytes_read = read(m_pipefd, buffer, sizeof(buffer) - 1);
            
            if (bytes_read > 0) {
                buffer[bytes_read] = '\0';
                std::cout << "Raw data from pipe: " << buffer << std::endl; // DEBUG LINE
                line_buffer += buffer;
                
                // Process complete lines (JSON messages end with newline)
                size_t pos;
                while ((pos = line_buffer.find('\n')) != std::string::npos) {
                    std::string json_line = line_buffer.substr(0, pos);
                    line_buffer.erase(0, pos + 1);
                    
                    if (!json_line.empty()) {
                        std::cout << "Processing line: " << json_line << std::endl; // DEBUG LINE
                        processGestureMessage(json_line);
                    }
                }
            } else if (bytes_read == 0) {
                // EOF - Python script closed the pipe
                std::cout << "Python script closed the pipe. Waiting for reconnection..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                
                // Try to reopen the pipe
                close(m_pipefd);
                m_pipefd = open(PIPE_PATH.c_str(), O_RDONLY);
                if (m_pipefd == -1) {
                    std::cerr << "Failed to reopen pipe. Exiting..." << std::endl;
                    break;
                }
            } else {
                // Error reading from pipe
                if (m_isRunning) {
                    perror("Error reading from pipe");
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
        }
        std::cout << "Exiting processing loop." << std::endl;
    }

    // Process a JSON message received from Python
    void processGestureMessage(const std::string& json_str) {
        try {
            // Try to parse as JSON first
            json data;
            try {
                data = json::parse(json_str);
            } catch (const json::parse_error& e) {
                // If not valid JSON, try to interpret as a simple command string
                // This is to handle the case where Python just sends "two_up" instead of proper JSON
                std::string command = json_str;
                // Trim whitespace
                command.erase(0, command.find_first_not_of(" \t\r\n"));
                command.erase(command.find_last_not_of(" \t\r\n") + 1);
                
                // Create a JSON object manually
                data = {
                    {"type", "gesture"},
                    {"command", command}
                };
                
                std::cout << "Converted plain text to JSON: " << data.dump() << std::endl;
            }
            
            // Now process the JSON message
            if (data.contains("type") && data["type"] == "gesture") {
                std::string command = data.value("command", "");
                CommandType cmd_type = m_webSocketClient.stringToCommandType(command);
                
                if (cmd_type != CommandType::UNKNOWN) {
                    std::cout << "Received gesture: " << command << std::endl;
                    
                    // Handle position data for tracking commands
                    json position_data;
                    if (data.contains("position")) {
                        position_data = data["position"];
                    }
                    
                    // Send command to WebSocket server
                    if (m_webSocketClient.isConnected()) {
                        bool sent = m_webSocketClient.sendCommand(cmd_type, position_data);
                        if (!sent) {
                            std::cerr << "Failed to send command: " << command << std::endl;
                        } else {
                            std::cout << "Successfully sent command to server: " << command << std::endl;
                        }
                    } else {
                        std::cout << "WebSocket not connected. Skipping command: " << command << std::endl;
                    }
                } else {
                    std::cout << "Unknown gesture command: " << command << std::endl;
                }
            } else if (!data.contains("type")) {
                // If there's no type field but it parsed as JSON, try to extract a command field
                if (data.contains("command")) {
                    std::string command = data["command"];
                    CommandType cmd_type = m_webSocketClient.stringToCommandType(command);
                    
                    if (cmd_type != CommandType::UNKNOWN) {
                        std::cout << "Received direct command JSON: " << command << std::endl;
                        
                        // Handle position data for tracking commands
                        json position_data;
                        if (data.contains("position")) {
                            position_data = data["position"];
                        }
                        
                        // Send command to WebSocket server
                        if (m_webSocketClient.isConnected()) {
                            bool sent = m_webSocketClient.sendCommand(cmd_type, position_data);
                            if (!sent) {
                                std::cerr << "Failed to send command: " << command << std::endl;
                            } else {
                                std::cout << "Successfully sent command to server: " << command << std::endl;
                            }
                        } else {
                            std::cout << "WebSocket not connected. Skipping command: " << command << std::endl;
                        }
                    } else {
                        std::cout << "Unknown gesture command: " << command << std::endl;
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error processing message: " << e.what() << std::endl;
            std::cerr << "Message was: " << json_str << std::endl;
        }
    }

    WebSocketHardwareClient    m_webSocketClient;  // Underlying WS client
    std::thread                m_processingThread; // Thread for the loop
    std::atomic<bool>          m_isRunning;        // Loop control flag
    int                        m_pipefd;           // File descriptor for named pipe
};

int main(int argc, char* argv[]) {
    // Default server URI and hardware client ID
    std::string serverUri = "ws://localhost:8080";
    std::string clientId  = "hardware-pi-01";

    // Override defaults via command-line arguments
    if (argc > 1) serverUri = argv[1];
    if (argc > 2) clientId  = argv[2];

    std::cout << "--- AirClass Hardware Client ---" << std::endl;
    std::cout << "Server URI: " << serverUri << std::endl;
    std::cout << "Client ID : " << clientId << std::endl;
    std::cout << "Named Pipe: " << PIPE_PATH << std::endl;

    // Instantiate and initialize the gesture system
    GestureControlSystem gestureSystem(serverUri, clientId);
    if (!gestureSystem.initialize()) {
        std::cerr << "FATAL: Could not initialize hardware client. Exiting." << std::endl;
        return 1;
    }

    // Begin gesture processing
    gestureSystem.start();
    

    
    // Wait for user input to terminate
    std::cout << "Hardware client running. Press Enter to exit." << std::endl;
    std::cin.get();

    std::cout << "Shutdown requested..." << std::endl;
    gestureSystem.stop();
    std::cout << "Hardware client finished." << std::endl;
    return 0;
}
