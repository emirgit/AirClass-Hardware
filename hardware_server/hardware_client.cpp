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

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

// For convenience
using json = nlohmann::json;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;

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

        m_client.clear_access_channels(websocketpp::log::alevel::all);
        m_client.set_access_channels(websocketpp::log::alevel::connect);
        m_client.set_access_channels(websocketpp::log::alevel::disconnect);
        m_client.set_access_channels(websocketpp::log::alevel::app);

        m_client.init_asio();

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
            auto con = m_client.get_connection(m_uri, ec);
            if (ec) {
                std::cerr << "Could not create connection: " << ec.message() << std::endl;
                return false;
            }

            m_hdl = con->get_handle();
            m_client.connect(con);

            m_thread = std::thread([this]() { 
                try {
                    m_client.run();
                } catch (const std::exception& e) {
                    std::cerr << "Exception in WebSocket run loop: " << e.what() << std::endl;
                }
            });

            std::unique_lock<std::mutex> lock(m_mutex);
            if (!m_cv.wait_for(lock, std::chrono::seconds(5), [this]{ return m_connected || m_reconnectAttempts > 0; })) {
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

        m_client.stop();
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
            std::string commandStr = commandTypeToString(command);
            json message = {
                {"type", "gesture_command"},
                {"source", "hardware"},
                {"client_id", m_clientId},
                {"command", commandStr},
                {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()}
            };

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

private:
    void on_open(websocketpp::connection_hdl hdl) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_connected = true;
            m_reconnectAttempts = 0;
        }

        std::cout << "Connection opened" << std::endl;

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

        if (m_reconnectAttempts < m_maxReconnectAttempts) {
            m_reconnectAttempts++;
            std::cout << "Attempting reconnect " << m_reconnectAttempts 
                      << " of " << m_maxReconnectAttempts << " in " 
                      << m_reconnectDelayMs << "ms..." << std::endl;

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

        if (m_reconnectAttempts < m_maxReconnectAttempts) {
            m_reconnectAttempts++;
            std::cout << "Attempting reconnect " << m_reconnectAttempts 
                      << " of " << m_maxReconnectAttempts << " in " 
                      << m_reconnectDelayMs << "ms..." << std::endl;

            std::thread([this]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(m_reconnectDelayMs));
                connect();
            }).detach();
        }

        m_cv.notify_one();
    }

    void on_message(websocketpp::connection_hdl hdl, message_ptr msg) {
        try {
            json data = json::parse(msg->get_payload());
            std::cout << "Received message: " << data.dump(2) << std::endl;
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

// =================== GestureControlSystem =====================

class GestureControlSystem {
public:
    GestureControlSystem(const std::string& serverUri, const std::string& clientId)
        : m_webSocketClient(serverUri, clientId), m_isRunning(false) {}

    ~GestureControlSystem() {
        stop();
    }

    bool initialize() {
        return m_webSocketClient.connect();
    }

    bool start() {
        if (!m_webSocketClient.isConnected()) {
            std::cerr << "Cannot start: WebSocket not connected" << std::endl;
            return false;
        }
        m_isRunning = true;
        m_tcpThread = std::thread(&GestureControlSystem::tcpListenerLoop, this);
        return true;
    }

    void stop() {
        m_isRunning = false;
        if (m_tcpThread.joinable()) {
            m_tcpThread.join();
        }
    }

private:
    void tcpListenerLoop() {
        int server_fd, new_socket;
        struct sockaddr_in address;
        int opt = 1;
        int addrlen = sizeof(address);
        char buffer[1024] = {0};

        server_fd = socket(AF_INET, SOCK_STREAM, 0);
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(65432);

        if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
            std::cerr << "TCP bind failed" << std::endl;
            return;
        }
        if (listen(server_fd, 3) < 0) {
            std::cerr << "TCP listen failed" << std::endl;
            return;
        }

        std::cout << "TCP server listening on port 65432" << std::endl;

        while (m_isRunning && (new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) >= 0) {
            std::cout << "Client connected to TCP server" << std::endl;
            int valread;
            while (m_isRunning && (valread = read(new_socket, buffer, 1024)) > 0) {
                buffer[valread] = '\0';
                std::string cmd(buffer);
                std::cout << "Received from Python: " << cmd << std::endl;

                if (cmd.find("next_slide") != std::string::npos) {
                    m_webSocketClient.sendCommand(CommandType::NEXT_SLIDE);
                } else if (cmd.find("previous_slide") != std::string::npos) {
                    m_webSocketClient.sendCommand(CommandType::PREVIOUS_SLIDE);
                }
            }
            close(new_socket);
            std::cout << "TCP client disconnected" << std::endl;
        }

        close(server_fd);
    }

    WebSocketHardwareClient m_webSocketClient;
    std::thread m_tcpThread;
    bool m_isRunning;
};

// =================== MAIN =====================

int main(int argc, char* argv[]) {
    std::string serverUri = "ws://localhost:8080";
    std::string clientId = "raspberry-pi-gesture";

    if (argc > 1) { serverUri = argv[1]; }
    if (argc > 2) { clientId = argv[2]; }

    try {
        std::cout << "Starting Gesture Control Hardware Client" << std::endl;
        std::cout << "Server URI: " << serverUri << std::endl;
        std::cout << "Client ID: " << clientId << std::endl;

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
