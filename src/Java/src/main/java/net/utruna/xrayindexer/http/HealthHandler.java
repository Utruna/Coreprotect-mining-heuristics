package net.utruna.xrayindexer.http;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import net.utruna.xrayindexer.db.ReadOnlyCoreProtectDb;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

/**
 * GET /health -> petit JSON { "status", "max_rowid" }.
 * Sert de sonde et donne au client la tête courante de co_block (pour savoir
 * s'il est à jour sans télécharger un lot).
 */
final class HealthHandler implements HttpHandler {

    private final ReadOnlyCoreProtectDb db;
    private final String blockTable;

    HealthHandler(ReadOnlyCoreProtectDb db, String blockTable) {
        this.db = db;
        this.blockTable = blockTable;
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
        String json;
        int code;
        try (Connection connection = db.open();
             Statement st = connection.createStatement();
             ResultSet rs = st.executeQuery("SELECT MAX(rowid) FROM " + blockTable + ";")) {
            long maxRowid = rs.next() ? rs.getLong(1) : 0L;
            json = "{\"status\":\"ok\",\"max_rowid\":" + maxRowid + "}\n";
            code = 200;
        } catch (SQLException e) {
            json = "{\"status\":\"error\",\"message\":" + jsonString(e.getMessage()) + "}\n";
            code = 500;
        }

        byte[] body = json.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(code, body.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(body);
        }
    }

    private static String jsonString(String s) {
        if (s == null) return "null";
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }
}
