// Include WebSocket++ headers (ASIO transport, no TLS for simplicity)
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

// Standard Library includes
#include <iostream>
#include <map>
#include <string>
#include <memory>
#include <mutex>
#include <thread>
#include <chrono>
#include <functional>
#include <cstdlib>
#include <stdexcept>

// Networking includes for UDP broadcasting
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

// JSON library for message parsing/serialization
#include <nlohmann/json.hpp>

// Convenient aliases
using json = nlohmann::json;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::connection_hdl;

// Define our server type using WebSocket++ with ASIO
typedef websocketpp::server<websocketpp::config::asio> server;
typedef server::message_ptr message_ptr;

// Simplified enumeration of client roles - we only care about hardware and desktop
enum class ClientType {
    HARDWARE,
    DESKTOP,
    UNKNOWN // Before the client registers
};

// Stores per-connection metadata
struct ClientInfo {
    ClientType type = ClientType::UNKNOWN;  // Role of this client
    std::string id = "";                    // Client-provided unique ID
};

// Function to discover desktop client via UDP broadcast
// Returns the IP address of the desktop if found, empty string otherwise
std::string discoverDesktopIP(int port = 9999, const std::string& broadcastMessage = "raspberry_discovery") {
    std::cout << "[UDP] Starting desktop discovery process..." << std::endl;
    
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("socket");
        return "";
    }

    // Enable broadcasting on the socket
    int broadcastEnable = 1;
    if (setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcastEnable, sizeof(broadcastEnable)) < 0) {
        perror("setsockopt");
        close(sock);
        return "";
    }

    // Set up broadcast address
    sockaddr_in broadcastAddr{};
    broadcastAddr.sin_family = AF_INET;
    broadcastAddr.sin_port = htons(port);
    broadcastAddr.sin_addr.s_addr = inet_addr("255.255.255.255");

    // Set up structures for receiving response
    sockaddr_in fromAddr{};
    socklen_t fromLen = sizeof(fromAddr);
    char buffer[128];

    // Try to discover desktop with multiple attempts
    const int MAX_ATTEMPTS = 5;
    for (int attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
        // Send discovery message
        ssize_t sent = sendto(sock, broadcastMessage.c_str(), broadcastMessage.length(), 0,
                             (sockaddr*)&broadcastAddr, sizeof(broadcastAddr));
        if (sent < 0) {
            perror("sendto");
        } else {
            std::cout << "[UDP] Broadcast sent (attempt " << attempt << "/" << MAX_ATTEMPTS 
                      << "), waiting for response..." << std::endl;
        }

        // Set 2-second receive timeout
        struct timeval tv = {2, 0};
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        // Try receiving response
        ssize_t recvLen = recvfrom(sock, buffer, sizeof(buffer) - 1, 0,
                                  (sockaddr*)&fromAddr, &fromLen);
        if (recvLen > 0) {
            buffer[recvLen] = '\0';
            std::string desktopIp(buffer);

            // Check if received data looks like a valid IP (basic check)
            if (!desktopIp.empty() && desktopIp.find('.') != std::string::npos) {
                std::cout << "[UDP] Desktop IP address found: " << desktopIp << std::endl;
                close(sock);
                return desktopIp;
            } else {
                std::cerr << "[UDP] Invalid IP response received: '" << desktopIp << "'" << std::endl;
            }
        } else {
            std::cout << "[UDP] No response received, trying again..." << std::endl;
        }

        // Wait before next attempt
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "[UDP] Desktop discovery failed after " << MAX_ATTEMPTS << " attempts" << std::endl;
    close(sock);
    return "";
}

class AirClassServer {
public:
    AirClassServer() {
        // Configure WebSocket++ logging: only show connect/disconnect/app logs
        m_server.clear_access_channels(websocketpp::log::alevel::all);
        m_server.set_access_channels(websocketpp::log::alevel::connect);
        m_server.set_access_channels(websocketpp::log::alevel::disconnect);
        m_server.set_access_channels(websocketpp::log::alevel::app);

        // Initialize ASIO subsystem
        m_server.init_asio();

        // Register callback handlers for lifecycle events
        m_server.set_open_handler(bind(&AirClassServer::on_open, this, _1));
        m_server.set_close_handler(bind(&AirClassServer::on_close, this, _1));
        m_server.set_message_handler(bind(&AirClassServer::on_message, this, _1, _2));
    }

    // Starts listening on the given port and runs the ASIO loop
    void run(uint16_t port) {
        try {
            m_server.listen(port);        // Bind socket to port
            m_server.start_accept();      // Begin accepting connections
            std::cout << "WebSocket Server started on port " << port << std::endl;
            
            // Try to discover desktop IP before entering event loop
            std::string desktopIp = discoverDesktopIP();
            if (!desktopIp.empty()) {
                std::cout << "Desktop client found at: " << desktopIp << std::endl;
                // You could store this IP for later use if needed
            }
            
            m_server.run();               // Enter the event loop (blocks)
        } catch (const websocketpp::exception& e) {
            std::cerr << "WebSocket Exception: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Standard Exception during run: " << e.what() << std::endl;
        }
    }

    // Gracefully shuts down all connections and stops the server loop
    void stop() {
        std::lock_guard<std::mutex> guard(m_connection_lock);

        // Close each open connection with a "going_away" status
        for (auto const& [hdl, info] : m_connections) {
            try {
                if (!hdl.expired()) {
                    m_server.close(hdl, websocketpp::close::status::going_away, "Server shutdown");
                }
            } catch (const websocketpp::exception& e) {
                std::cerr << "Exception closing connection: " << e.what() << std::endl;
            }
        }
        m_connections.clear();

        // Stop accepting new connections and exit the ASIO loop
        m_server.stop_listening();
        m_server.stop();
        std::cout << "WebSocket Server stopped." << std::endl;
    }

private:
    // Handler: new client connection opened
    void on_open(connection_hdl hdl) {
        std::lock_guard<std::mutex> guard(m_connection_lock);
        // Initialize ClientInfo with UNKNOWN type
        m_connections[hdl] = std::make_shared<ClientInfo>();
        std::cout << "Connection opened. Awaiting registration." << std::endl;
    }

    // Handler: client connection closed
    void on_close(connection_hdl hdl) {
        std::lock_guard<std::mutex> guard(m_connection_lock);
        auto it = m_connections.find(hdl);
        if (it != m_connections.end()) {
            // Log the disconnected client's type and ID
            std::cout << "Client disconnected: Type=" 
                      << clientTypeToString(it->second->type)
                      << ", ID=" << (it->second->id.empty() ? "[unregistered]" : it->second->id)
                      << std::endl;
            m_connections.erase(it);
        } else {
            std::cout << "Connection closed (already removed or unknown)." << std::endl;
        }
    }

    // Handler: message received from a client
    void on_message(connection_hdl hdl, message_ptr msg) {
        std::shared_ptr<ClientInfo> sender_info;
        ClientType sender_type = ClientType::UNKNOWN;
        std::string payload = msg->get_payload();  // Extract raw message

        {   // Retrieve sender metadata under lock
            std::lock_guard<std::mutex> guard(m_connection_lock);
            auto it = m_connections.find(hdl);
            if (it == m_connections.end()) {
                std::cerr << "Error: Message from unknown connection." << std::endl;
                return;
            }
            sender_info = it->second;
            sender_type = sender_info->type;
        }

        // 1) If not yet registered, handle registration flow
        if (sender_type == ClientType::UNKNOWN) {
            handle_registration(hdl, sender_info, payload);
            return;
        }

        // 2) For hardware clients, print detailed JSON data
        if (sender_type == ClientType::HARDWARE) {
            std::cout << "\n╔══════════════════════════════════════════════╗" << std::endl;
            std::cout << "║ HARDWARE MESSAGE RECEIVED                     ║" << std::endl;
            std::cout << "╠══════════════════════════════════════════════╣" << std::endl;
            std::cout << "║ Client ID: " << sender_info->id << std::endl;
            
            // Try to parse and pretty-print the JSON
            try {
                json data = json::parse(payload);
                
                // Extract and display command if available
                if (data.contains("command")) {
                    std::cout << "║ Command: " << data["command"] << std::endl;
                }
                
                // Check for position data
                if (data.contains("position")) {
                    std::cout << "║ Position data: ";
                    auto position = data["position"];
                    // If position is an object with coordinates
                    if (position.is_object()) {
                        if (position.contains("x")) std::cout << "x=" << position["x"] << " ";
                        if (position.contains("y")) std::cout << "y=" << position["y"] << " ";
                        if (position.contains("z")) std::cout << "z=" << position["z"] << " ";
                    }
                    std::cout << std::endl;
                }
                
                // Print the full JSON payload for reference
                std::cout << "║ Raw JSON: " << payload << std::endl;
            }
            catch (const json::parse_error& e) {
                std::cout << "║ [Error parsing JSON] Raw payload: " << payload << std::endl;
                std::cout << "║ Parse error: " << e.what() << std::endl;
            }
            
            std::cout << "╚══════════════════════════════════════════════╝\n" << std::endl;
            
            // Forward from hardware to all desktop clients
            forward_message_to_desktops(payload, msg->get_opcode());
        } 
        else if (sender_type == ClientType::DESKTOP) {
            // For desktop clients, just log receipt
            std::cout << "Message from desktop client (ID: " << sender_info->id 
                      << ") received but not forwarded." << std::endl;
        }
    }

    // Parses and validates client registration messages
    void handle_registration(connection_hdl hdl,
                             std::shared_ptr<ClientInfo> client_info,
                             const std::string& payload) {
        try {
            json data = json::parse(payload);

            // Check required fields
            if (!data.contains("register") || !data.contains("id")) {
                send_error(hdl, "Registration requires 'register' and 'id'.");
                return;
            }

            std::string type_str = data["register"];
            std::string client_id = data["id"];
            if (client_id.empty()) {
                send_error(hdl, "Client ID cannot be empty.");
                return;
            }

            // Map string to enum - simplified to only care about hardware and desktop
            ClientType new_type = ClientType::UNKNOWN;
            if (type_str == "hardware") new_type = ClientType::HARDWARE;
            else if (type_str == "desktop") new_type = ClientType::DESKTOP;
            else {
                send_error(hdl, "Invalid client type: " + type_str + ". Must be 'hardware' or 'desktop'.");
                return;
            }

            // Update metadata and confirm registration
            client_info->type = new_type;
            client_info->id = client_id;
            std::cout << "Client registered: Type=" 
                      << clientTypeToString(new_type)
                      << ", ID=" << client_id << std::endl;

            json confirmation = {
                {"type", "registration_success"},
                {"client_type", clientTypeToString(new_type)},
                {"client_id", client_id}
            };
            m_server.send(hdl, confirmation.dump(), websocketpp::frame::opcode::text);

        } catch (const json::parse_error& e) {
            send_error(hdl, "Invalid JSON for registration.");
        } catch (const std::exception& e) {
            std::cerr << "Registration error: " << e.what() << std::endl;
            send_error(hdl, "Internal server error.");
        }
    }

    // Broadcasts a message payload to all desktop clients
    void forward_message_to_desktops(const std::string& payload,
                                    websocketpp::frame::opcode::value opcode) {
        std::lock_guard<std::mutex> guard(m_connection_lock);
        int sent_count = 0;

        try {
            // Try to parse the payload as JSON for better logging
            json data = json::parse(payload);
            std::cout << "Forwarding JSON message: " << std::endl;
            std::cout << "  Command: " << (data.contains("command") ? data["command"].dump() : "N/A") << std::endl;
            if (data.contains("position")) {
                std::cout << "  Position: " << data["position"].dump() << std::endl;
            }
        } catch (const json::parse_error& e) {
            // Not JSON or invalid JSON
            std::cout << "Forwarding non-JSON message: " << payload << std::endl;
        }

        for (auto const& [hdl, info] : m_connections) {
            if (info->type == ClientType::DESKTOP) {
                try {
                    if (!hdl.expired()) {
                        m_server.send(hdl, payload, opcode);
                        sent_count++;
                    }
                } catch (const websocketpp::exception& e) {
                    std::cerr << "Send error to desktop ID " << info->id << ": " << e.what() << std::endl;
                }
            }
        }
        
        if (sent_count > 0) {
            std::cout << "Successfully forwarded message to " << sent_count << " desktop client(s)" << std::endl;
        } else {
            std::cout << "No desktop clients connected to forward message to" << std::endl;
        }
    }

    // Sends a structured error JSON to a single client
    void send_error(connection_hdl hdl, const std::string& error_message) {
        json error_json = {
            {"type", "error"},
            {"message", error_message}
        };
        try {
            if (!hdl.expired()) {
                m_server.send(hdl, error_json.dump(), websocketpp::frame::opcode::text);
            }
        } catch (const websocketpp::exception& e) {
            std::cerr << "Failed to send error: " << e.what() << std::endl;
        }
    }

    // Utility: convert ClientType enum to a readable string
    std::string clientTypeToString(ClientType type) {
        switch (type) {
            case ClientType::HARDWARE: return "Hardware";
            case ClientType::DESKTOP:  return "Desktop";
            case ClientType::UNKNOWN:  return "Unknown";
        }
        return "InvalidType";
    }

    server m_server;  // Underlying WebSocket++ server instance
    // Maps connection handles to their associated client metadata
    std::map<connection_hdl, std::shared_ptr<ClientInfo>, std::owner_less<connection_hdl>> m_connections;
    std::mutex m_connection_lock;  // Protects m_connections across threads
};

int main(int argc, char* argv[]) {
    uint16_t port = 8080;  // Default listening port
    
    // Attempt to read Render.com-provided PORT env var first
    if (const char* env_port = std::getenv("PORT")) {
        try {
            port = std::stoi(env_port);
        } catch (...) {
            std::cerr << "Warning: Invalid PORT env var, using default " << port << "." << std::endl;
        }
    }
    // Otherwise, fall back to a CLI argument if provided
    else if (argc > 1) {
        try {
            port = std::stoi(argv[1]);
        } catch (...) {
            std::cerr << "Warning: Invalid port arg, using default " << port << "." << std::endl;
        }
    }

    std::cout << "--- AirClass Server ---" << std::endl;
    std::cout << "Starting WebSocket server on port " << port << std::endl;
    
    AirClassServer server_instance;
    server_instance.run(port);  // Start the server event loop
    
    return 0;
}
