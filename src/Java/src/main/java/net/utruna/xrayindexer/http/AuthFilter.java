package net.utruna.xrayindexer.http;

import com.sun.net.httpserver.Filter;
import com.sun.net.httpserver.HttpExchange;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

/**
 * Exige un en-tête « Authorization: Bearer &lt;token&gt; » sur chaque requête.
 * La passerelle expose des données du serveur : sans jeton, pas d'accès.
 * Comparaison à temps constant pour ne pas fuir le jeton par timing.
 */
final class AuthFilter extends Filter {

    private final byte[] expected;

    AuthFilter(String token) {
        this.expected = token.getBytes(StandardCharsets.UTF_8);
    }

    @Override
    public String description() {
        return "Authentification par jeton Bearer.";
    }

    @Override
    public void doFilter(HttpExchange exchange, Chain chain) throws IOException {
        String header = exchange.getRequestHeaders().getFirst("Authorization");
        if (header == null || !header.startsWith("Bearer ")) {
            reject(exchange);
            return;
        }
        byte[] provided = header.substring("Bearer ".length()).trim()
                .getBytes(StandardCharsets.UTF_8);
        if (!MessageDigest.isEqual(provided, expected)) {
            reject(exchange);
            return;
        }
        chain.doFilter(exchange);
    }

    private void reject(HttpExchange exchange) throws IOException {
        byte[] body = "401 Unauthorized\n".getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("WWW-Authenticate", "Bearer");
        exchange.sendResponseHeaders(401, body.length);
        exchange.getResponseBody().write(body);
        exchange.close();
    }
}
