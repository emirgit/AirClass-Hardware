#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/client.hpp>
#include <iostream>
#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <nlohmann/json.hpp>

// For convenience
using json = nlohmann::json;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;

// Define client type with asio (no TLS)
typedef websocketpp::client<websocketpp::config::asio> websocket_client;
typedef websocketpp::config::asio::message_type::ptr message_ptr;

// Enum for command types
enum class CommandType {
    NEXT_SLIDE,
    PREVIOUS_SLIDE,
    ZOOM_IN,
    ZOOM_OUT,
    START_PRESENTATION,
    END_PRESENTATION,
    GRANT_PERMISSION,
    DENY_PERMISSION,
    NO_COMMAND
};

class WebSocketHardwareClient {
public:
    WebSocketHardwareClient(const std::string& uri, const std::string& clientId) 
        : m_uri(uri), m_clientId(clientId), m_connected(false), m_reconnectAttempts(0), 
          m_maxReconnectAttempts(5), m_reconnectDelayMs(2000) {
        
        // Set logging settings
        m_client.clear_access_channels(websocketpp::log::alevel::all);
        m_client.set_access_channels(websocketpp::log::alevel::connect);
        m_client.set_access_channels(websocketpp::log::alevel::disconnect);
        m_client.set_access_channels(websocketpp::log::alevel::app);
        
        // Initialize ASIO
        m_client.init_asio();
        
        // Set handlers
        m_client.set_open_handler(bind(&WebSocketHardwareClient::on_open, this, _1));
        m_client.set_close_handler(bind(&WebSocketHardwareClient::on_close, this, _1));
        m_client.set_fail_handler(bind(&WebSocketHardwareClient::on_fail, this, _1));
        m_client.set_message_handler(bind(&WebSocketHardwareClient::on_message, this, _1, _2));
    }
    
    ~WebSocketHardwareClient() {
        disconnect();
    }
    
    bool connect() {
        try {
            websocketpp::lib::error_code ec;
            
            // Create connection
            auto con = m_client.get_connection(m_uri, ec);
            if (ec) {
                std::cerr << "Could not create connection: " << ec.message() << std::endl;
                return false;
            }
            
            // Store the connection handle
            m_hdl = con->get_handle();
            
            // Connect to server
            m_client.connect(con);
            
            // Start the ASIO io_service run loop in a separate thread
            m_thread = std::thread([this]() { 
                try {
                    m_client.run();
                } catch (const std::exception& e) {
                    std::cerr << "Exception in WebSocket run loop: " << e.what() << std::endl;
                }
            });
            
            // Wait for connection to be established or fail
            std::unique_lock<std::mutex> lock(m_mutex);
            if (!m_cv.wait_for(lock, std::chrono::seconds(5), 
                                [this]{ return m_connected || m_reconnectAttempts > 0; })) {
                std::cerr << "Connection timeout" << std::endl;
                return false;
            }
            
            return m_connected;
        }
        catch (const std::exception& e) {
            std::cerr << "Exception during connect: " << e.what() << std::endl;
            return false;
        }
    }
    
    void disconnect() {
        if (m_connected) {
            try {
                websocketpp::lib::error_code ec;
                m_client.close(m_hdl, websocketpp::close::status::normal, "Closing connection", ec);
                
                if (ec) {
                    std::cerr << "Error closing connection: " << ec.message() << std::endl;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Exception during disconnect: " << e.what() << std::endl;
            }
        }
        
        // Stop the client
        m_client.stop();
        
        // Join the thread if joinable
        if (m_thread.joinable()) {
            m_thread.join();
        }
        
        m_connected = false;
    }
    
    bool sendCommand(CommandType command) {
        if (!m_connected) {
            std::cerr << "Not connected to server" << std::endl;
            return false;
        }
        
        try {
            // Convert command enum to string
            std::string commandStr = commandTypeToString(command);
            
            // Create JSON message
            json message = {
                {"type", "gesture_command"},
                {"source", "hardware"},
                {"client_id", m_clientId},
                {"command", commandStr},
                {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()}
            };
            
            // Send the message
            websocketpp::lib::error_code ec;
            m_client.send(m_hdl, message.dump(), websocketpp::frame::opcode::text, ec);
            
            if (ec) {
                std::cerr << "Error sending message: " << ec.message() << std::endl;
                return false;
            }
            
            std::cout << "Sent command: " << commandStr << std::endl;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Exception during sendCommand: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool isConnected() const {
        return m_connected;
    }
    
    void setReconnectParameters(int maxAttempts, int delayMs) {
        m_maxReconnectAttempts = maxAttempts;
        m_reconnectDelayMs = delayMs;
    }

private:
    void on_open(websocketpp::connection_hdl hdl) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_connected = true;
            m_reconnectAttempts = 0;
        }
        
        std::cout << "Connection opened" << std::endl;
        
        // Register as hardware client
        json registration = {
            {"register", "hardware"},
            {"id", m_clientId}
        };
        
        websocketpp::lib::error_code ec;
        m_client.send(hdl, registration.dump(), websocketpp::frame::opcode::text, ec);
        
        if (ec) {
            std::cerr << "Error registering client: " << ec.message() << std::endl;
        }
        
        m_cv.notify_one();
    }
    
    void on_close(websocketpp::connection_hdl hdl) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_connected = false;
        }
        
        auto conn = m_client.get_con_from_hdl(hdl);
        std::cout << "Connection closed: " << conn->get_ec().message() << std::endl;
        
        // Attempt to reconnect
        if (m_reconnectAttempts < m_maxReconnectAttempts) {
            m_reconnectAttempts++;
            std::cout << "Attempting reconnect " << m_reconnectAttempts 
                      << " of " << m_maxReconnectAttempts << " in " 
                      << m_reconnectDelayMs << "ms..." << std::endl;
            
            // Schedule reconnect after delay
            std::thread([this]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(m_reconnectDelayMs));
                connect();
            }).detach();
        }
        
        m_cv.notify_one();
    }
    
    void on_fail(websocketpp::connection_hdl hdl) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_connected = false;
        }
        
        auto conn = m_client.get_con_from_hdl(hdl);
        std::cout << "Connection failed: " << conn->get_ec().message() << std::endl;
        
        // Attempt to reconnect (same as on_close)
        if (m_reconnectAttempts < m_maxReconnectAttempts) {
            m_reconnectAttempts++;
            std::cout << "Attempting reconnect " << m_reconnectAttempts 
                      << " of " << m_maxReconnectAttempts << " in " 
                      << m_reconnectDelayMs << "ms..." << std::endl;
            
            // Schedule reconnect after delay
            std::thread([this]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(m_reconnectDelayMs));
                connect();
            }).detach();
        }
        
        m_cv.notify_one();
    }
    
    void on_message(websocketpp::connection_hdl hdl, message_ptr msg) {
        try {
            // Parse JSON message
            json data = json::parse(msg->get_payload());
            
            std::cout << "Received message: " << data.dump(2) << std::endl;
            
            // Process different message types
            if (data.contains("type")) {
                std::string type = data["type"];
                
                if (type == "registration_success") {
                    std::cout << "Successfully registered as hardware client with ID: " 
                              << data["client_id"] << std::endl;
                }
                else if (type == "error") {
                    std::cerr << "Error from server: " << data["message"] << std::endl;
                }
                // Add handlers for other message types as needed
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error processing message: " << e.what() << std::endl;
        }
    }
    
    std::string commandTypeToString(CommandType command) {
        switch (command) {
            case CommandType::NEXT_SLIDE: return "next_slide";
            case CommandType::PREVIOUS_SLIDE: return "previous_slide";
            case CommandType::ZOOM_IN: return "zoom_in";
            case CommandType::ZOOM_OUT: return "zoom_out";
            case CommandType::START_PRESENTATION: return "start_presentation";
            case CommandType::END_PRESENTATION: return "end_presentation";
            case CommandType::GRANT_PERMISSION: return "grant_permission";
            case CommandType::DENY_PERMISSION: return "deny_permission";
            default: return "unknown";
        }
    }

    websocket_client m_client;
    websocketpp::connection_hdl m_hdl;
    std::thread m_thread;
    std::string m_uri;
    std::string m_clientId;
    bool m_connected;
    int m_reconnectAttempts;
    int m_maxReconnectAttempts;
    int m_reconnectDelayMs;
    std::mutex m_mutex;
    std::condition_variable m_cv;
};

// Simple integration example with the main hardware controller system
class GestureControlSystem {
public:
    GestureControlSystem(const std::string& serverUri, const std::string& clientId)
        : m_webSocketClient(serverUri, clientId), m_isRunning(false) {
    }
    
    ~GestureControlSystem() {
        stop();
    }
    
    bool initialize() {
        // Initialize components here (camera, gesture detector, etc.)
        // For this example, we'll just connect to the WebSocket server
        
        return m_webSocketClient.connect();
    }
    
    bool start() {
        if (!m_webSocketClient.isConnected()) {
            std::cerr << "Cannot start: WebSocket not connected" << std::endl;
            return false;
        }
        
        m_isRunning = true;
        
        // Start the processing thread
        m_processingThread = std::thread(&GestureControlSystem::processingLoop, this);
        
        return true;
    }
    
    void stop() {
        m_isRunning = false;
        
        if (m_processingThread.joinable()) {
            m_processingThread.join();
        }
    }

private:
    void processingLoop() {
        // Simulate gesture detection and command sending
        int counter = 0;
        
        while (m_isRunning) {
            // In a real system, this would detect actual gestures
            // For demonstration, we'll alternate between commands
            if (counter % 5 == 0) {
                m_webSocketClient.sendCommand(CommandType::NEXT_SLIDE);
            }
            else if (counter % 5 == 2) {
                m_webSocketClient.sendCommand(CommandType::PREVIOUS_SLIDE);
            }
            
            counter++;
            
            // Sleep to simulate processing time
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    }
    
    WebSocketHardwareClient m_webSocketClient;
    std::thread m_processingThread;
    bool m_isRunning;
};

// Main function to demonstrate usage
int main(int argc, char* argv[]) {
    // Default server URI and client ID
    std::string serverUri = "ws://localhost:8080";
    std::string clientId = "raspberry-pi-gesture";
    
    // Allow server URI to be specified as command line argument
    if (argc > 1) {
        serverUri = argv[1];
    }
    
    // Allow client ID to be specified as command line argument
    if (argc > 2) {
        clientId = argv[2];
    }
    
    try {
        std::cout << "Starting Gesture Control Hardware Client" << std::endl;
        std::cout << "Server URI: " << serverUri << std::endl;
        std::cout << "Client ID: " << clientId << std::endl;
        
        // Create and initialize the gesture control system
        GestureControlSystem system(serverUri, clientId);
        
        if (!system.initialize()) {
            std::cerr << "Failed to initialize the system" << std::endl;
            return 1;
        }
        
        if (!system.start()) {
            std::cerr << "Failed to start the system" << std::endl;
            return 1;
        }
        
        std::cout << "System running. Press Enter to exit." << std::endl;
        std::cin.get();
        
        system.stop();
        std::cout << "System stopped" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
