#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/client.hpp>
#include <iostream>
#include <thread>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
// Define client with NO TLS
typedef websocketpp::client<websocketpp::config::asio> client;

int main() {
    // Create a client instance
    client c;
    
    // Turn off all logging
    c.clear_access_channels(websocketpp::log::alevel::all);
    c.set_access_channels(websocketpp::log::alevel::connect);
    c.set_access_channels(websocketpp::log::alevel::disconnect);
    
    // Initialize ASIO
    c.init_asio();
    
    // Set message handler
    c.set_message_handler([](websocketpp::connection_hdl hdl, client::message_ptr msg) {
        std::cout << "Received: " << msg->get_payload() << std::endl;
    });
    
    // Connect to the WebSocket server
    websocketpp::lib::error_code ec;
    auto conn = c.get_connection("ws://localhost:8080", ec);
    
    if (ec) {
        std::cout << "Connection error: " << ec.message() << std::endl;
        return 1;
    }
    
    c.connect(conn);
    
    // Run the ASIO io_service in a separate thread
    std::thread t([&c]() { 
        try {
            c.run();
        } catch (const std::exception& e) {
            std::cout << "Exception in thread: " << e.what() << std::endl;
        }
    });
    
    // Wait a moment for connection to establish
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // Register as desktop
    json reg = {
        {"register", "desktop"},
        {"id", "test-desktop-1"}
    };
    
    try {
        conn->send(reg.dump());
        std::cout << "Registered as desktop client" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Failed to send registration: " << e.what() << std::endl;
    }
    
    std::cout << "Press Enter to exit" << std::endl;
    std::cin.get();
    
    // Cleanup
    c.stop();
    if (t.joinable()) {
        t.join();
    }
    
    return 0;
}
