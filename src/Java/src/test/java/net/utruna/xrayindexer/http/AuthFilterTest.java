package net.utruna.xrayindexer.http;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.sun.net.httpserver.HttpServer;

import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class AuthFilterTest {

    private HttpServer server;
    private URI endpoint;

    @BeforeEach
    void startServer() throws Exception {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        var context = server.createContext("/protected", exchange -> {
            byte[] body = "ok".getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, body.length);
            exchange.getResponseBody().write(body);
            exchange.close();
        });
        context.getFilters().add(new AuthFilter("correct-token"));
        server.start();
        endpoint = URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/protected");
    }

    @AfterEach
    void stopServer() {
        server.stop(0);
    }

    @Test
    void rejectsMissingAuthorizationHeader() throws Exception {
        HttpResponse<String> response = send(null);

        assertEquals(401, response.statusCode());
        assertEquals("Bearer", response.headers().firstValue("WWW-Authenticate").orElseThrow());
        assertEquals("401 Unauthorized\n", response.body());
    }

    @Test
    void rejectsInvalidBearerToken() throws Exception {
        assertEquals(401, send("Bearer wrong-token").statusCode());
    }

    @Test
    void permitsExactBearerToken() throws Exception {
        HttpResponse<String> response = send("Bearer correct-token");

        assertEquals(200, response.statusCode());
        assertEquals("ok", response.body());
    }

    private HttpResponse<String> send(String authorization) throws Exception {
        HttpRequest.Builder request = HttpRequest.newBuilder(endpoint).GET();
        if (authorization != null) {
            request.header("Authorization", authorization);
        }
        return HttpClient.newHttpClient().send(request.build(), HttpResponse.BodyHandlers.ofString());
    }
}