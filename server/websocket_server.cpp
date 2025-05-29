// Include WebSocket++ headers (ASIO transport, no TLS for simplicity)
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

// Standard Library includes
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <functional>
#include <memory>
#include <mutex>
#include <cstdlib>
#include <stdexcept>

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

// Enumeration of client roles for routing logic
enum class ClientType {
    HARDWARE,
    DESKTOP,
    MOBILE,
    UNKNOWN // Before the client registers
};

// Stores per-connection metadata
struct ClientInfo {
    ClientType type = ClientType::UNKNOWN;  // Role of this client
    std::string id = "";                    // Client-provided unique ID
};

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

        // 2) Already registered: log incoming message
        std::cout << "Received message from " << clientTypeToString(sender_type)
                  << " (ID: " << sender_info->id << "): " << payload << std::endl;

        // Determine where to forward based on sender role
        ClientType target_type = ClientType::UNKNOWN;
        if (sender_type == ClientType::HARDWARE) {
            target_type = ClientType::DESKTOP;
        } else if (sender_type == ClientType::DESKTOP) {
            target_type = ClientType::MOBILE;
        } else if (sender_type == ClientType::MOBILE) {
            target_type = ClientType::DESKTOP;
        } else {
            std::cerr << "Warning: Unexpected sender type post-registration." << std::endl;
            return;
        }

        // 3) Forward the payload to all clients of the target type
        forward_message_to_type(payload, msg->get_opcode(), target_type);
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

            // Map string to enum
            ClientType new_type = ClientType::UNKNOWN;
            if (type_str == "hardware") new_type = ClientType::HARDWARE;
            else if (type_str == "desktop") new_type = ClientType::DESKTOP;
            else if (type_str == "mobile") new_type = ClientType::MOBILE;
            else {
                send_error(hdl, "Invalid client type: " + type_str);
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

    // Broadcasts a message payload to all clients of a given type
    void forward_message_to_type(const std::string& payload,
                                 websocketpp::frame::opcode::value opcode,
                                 ClientType target_type) {
        std::lock_guard<std::mutex> guard(m_connection_lock);
        int sent_count = 0;

        for (auto const& [hdl, info] : m_connections) {
            if (info->type == target_type) {
                try {
                    if (!hdl.expired()) {
                        m_server.send(hdl, payload, opcode);
                        sent_count++;
                    }
                } catch (const websocketpp::exception& e) {
                    std::cerr << "Send error to ID " << info->id << ": " << e.what() << std::endl;
                }
            }
        }
        std::cout << "Forwarded message to " << sent_count
                  << " client(s) of type " << clientTypeToString(target_type) << std::endl;
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
            case ClientType::MOBILE:   return "Mobile";
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

    AirClassServer server_instance;
    server_instance.run(port);  // Start the server event loop
    return 0;
}
