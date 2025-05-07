#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <functional>
#include <nlohmann/json.hpp>

// For convenience
using json = nlohmann::json;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::connection_hdl;

// Define server type with chosen config
typedef websocketpp::server<websocketpp::config::asio> server;

enum ClientType {
    HARDWARE,
    DESKTOP,
    MOBILE,
    UNKNOWN
};

struct ClientInfo {
    connection_hdl hdl;
    ClientType type;
    std::string id;
};

class ClassroomWebSocketServer {
public:
    ClassroomWebSocketServer() {
        // Set logging settings
        m_server.set_access_channels(websocketpp::log::alevel::all);
        m_server.clear_access_channels(websocketpp::log::alevel::frame_payload);

        // Initialize ASIO
        m_server.init_asio();

        // Set callback handlers
        m_server.set_open_handler(bind(&ClassroomWebSocketServer::on_open, this, _1));
        m_server.set_close_handler(bind(&ClassroomWebSocketServer::on_close, this, _1));
        m_server.set_message_handler(bind(&ClassroomWebSocketServer::on_message, this, _1, _2));
    }

    void run(uint16_t port) {
        // Listen on specified port
        m_server.listen(port);
        
        // Start the server accept loop
        m_server.start_accept();
        
        // Start the ASIO io_service run loop
        std::cout << "WebSocket Server started on port " << port << std::endl;
        m_server.run();
    }

private:
    void on_open(connection_hdl hdl) {
        std::cout << "New connection opened" << std::endl;
        
        // Add to connections set (unidentified at first)
        ClientInfo info;
        info.hdl = hdl;
        info.type = UNKNOWN;
        info.id = "";
        
        m_connections[hdl] = info;
    }

    void on_close(connection_hdl hdl) {
        std::cout << "Connection closed" << std::endl;
        
        auto it = m_connections.find(hdl);
        if (it != m_connections.end()) {
            if (it->second.type != UNKNOWN) {
                std::cout << "Client disconnected: " 
                          << clientTypeToString(it->second.type) 
                          << " ID: " << it->second.id << std::endl;
            }
            m_connections.erase(it);
        }
    }

    void on_message(connection_hdl hdl, server::message_ptr msg) {
        try {
            auto& client = m_connections[hdl];
            
            // Parse JSON message
            json data = json::parse(msg->get_payload());
            
            // Handle client registration if needed
            if (client.type == UNKNOWN && data.contains("register")) {
                registerClient(hdl, data);
                return;
            }
            
            // Process various message types
            if (data.contains("type")) {
                std::string msgType = data["type"];
                
                std::cout << "Received message type: " << msgType 
                          << " from " << clientTypeToString(client.type) 
                          << " ID: " << client.id << std::endl;
                
                // Process based on message type
                if (msgType == "gesture_command") {
                    // Forward gesture commands to desktop applications
                    forwardToDesktop(data);
                }
                else if (msgType == "attendance") {
                    // Forward attendance data to desktop applications
                    forwardToDesktop(data);
                }
                else if (msgType == "speak_request") {
                    // Forward speak requests to desktop applications
                    forwardToDesktop(data);
                }
                else if (msgType == "permission_grant") {
                    // Forward permission grants to specific mobile clients
                    if (data.contains("target_id")) {
                        forwardToSpecificMobile(data["target_id"], data);
                    }
                }
                else {
                    // Broadcast other messages to all clients except sender
                    broadcastMessage(msg->get_payload(), hdl);
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error processing message: " << e.what() << std::endl;
            
            // Send error response
            json error = {
                {"type", "error"},
                {"message", "Invalid message format"}
            };
            
            m_server.send(hdl, error.dump(), msg->get_opcode());
        }
    }
    
    void registerClient(connection_hdl hdl, const json& data) {
        auto& client = m_connections[hdl];
        
        if (data["register"] == "hardware") {
            client.type = HARDWARE;
            client.id = data.value("id", "hardware-" + std::to_string(m_nextId++));
        }
        else if (data["register"] == "desktop") {
            client.type = DESKTOP;
            client.id = data.value("id", "desktop-" + std::to_string(m_nextId++));
        }
        else if (data["register"] == "mobile") {
            client.type = MOBILE;
            client.id = data.value("id", "mobile-" + std::to_string(m_nextId++));
        }
        
        std::cout << "Client registered as " << clientTypeToString(client.type) 
                  << " with ID: " << client.id << std::endl;
        
        // Send confirmation
        json confirmation = {
            {"type", "registration_success"},
            {"client_type", clientTypeToString(client.type)},
            {"client_id", client.id}
        };
        
        m_server.send(hdl, confirmation.dump(), websocketpp::frame::opcode::text);
    }
    
    void forwardToDesktop(const json& message) {
        std::string msgStr = message.dump();
        
        for (auto& pair : m_connections) {
            if (pair.second.type == DESKTOP) {
                m_server.send(pair.second.hdl, msgStr, websocketpp::frame::opcode::text);
            }
        }
    }
    
    void forwardToSpecificMobile(const std::string& targetId, const json& message) {
        std::string msgStr = message.dump();
        
        for (auto& pair : m_connections) {
            if (pair.second.type == MOBILE && pair.second.id == targetId) {
                m_server.send(pair.second.hdl, msgStr, websocketpp::frame::opcode::text);
                return;
            }
        }
        
        std::cerr << "Target mobile client not found: " << targetId << std::endl;
    }
    
    void broadcastMessage(const std::string& message, connection_hdl excludeHdl) {
        for (auto& pair : m_connections) {
            // Don't send back to the sender
            if (pair.first.lock() != excludeHdl.lock()) {
                m_server.send(pair.second.hdl, message, websocketpp::frame::opcode::text);
            }
        }
    }
    
    std::string clientTypeToString(ClientType type) {
        switch (type) {
            case HARDWARE: return "Hardware";
            case DESKTOP: return "Desktop";
            case MOBILE: return "Mobile";
            default: return "Unknown";
        }
    }

    server m_server;
    std::map<connection_hdl, ClientInfo, std::owner_less<connection_hdl>> m_connections;
    int m_nextId = 1;
};

int main(int argc, char* argv[]) {
    // Default port
    uint16_t port = 8080;
    
    // Allow port to be specified as command line argument
    if (argc > 1) {
        port = std::stoi(argv[1]);
    }
    
    try {
        ClassroomWebSocketServer server;
        server.run(port);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
    
    return 0;
}
