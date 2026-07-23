package net.utruna.xrayindexer.http;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import net.utruna.xrayindexer.db.ReadOnlyCoreProtectDb;

import java.io.IOException;
import java.util.logging.Logger;

/**
 * Renvoie une table de correspondance complète (co_user, co_material_map,
 * co_world) en CSV gzippé. Ces tables sont petites et évoluent lentement : le
 * client les recharge entièrement à chaque synchro plutôt que de les suivre par
 * curseur. Le plugin ne fait, là encore, que servir des lignes brutes.
 */
final class MapHandler implements HttpHandler {

    private final ReadOnlyCoreProtectDb db;
    private final String sql;
    private final Logger log;

    MapHandler(ReadOnlyCoreProtectDb db, String sql, Logger log) {
        this.db = db;
        this.sql = sql;
        this.log = log;
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
        if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
            exchange.sendResponseHeaders(405, -1);
            exchange.close();
            return;
        }
        QueryResponder.respondCsvGzip(exchange, db, sql, ps -> {
        }, log);
    }
}
