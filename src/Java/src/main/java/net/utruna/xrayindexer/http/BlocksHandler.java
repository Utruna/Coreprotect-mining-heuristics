package net.utruna.xrayindexer.http;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import net.utruna.xrayindexer.db.ReadOnlyCoreProtectDb;

import java.io.IOException;
import java.util.Map;
import java.util.logging.Logger;

/**
 * GET /blocks?since=&lt;rowid&gt;&amp;limit=&lt;n&gt;
 *
 * Renvoie une tranche brute de co_block (colonnes scalaires uniquement, jamais
 * les BLOBs meta/blockdata), paginée par rowid croissant. Le client suit le
 * curseur : il repart du plus grand cp_rowid reçu jusqu'à obtenir un lot vide.
 *
 * Aucun filtre (action, uuid, posé-puis-recassé) n'est appliqué ici : la
 * passerelle n'est qu'une porte. Toute la logique d'analyse vit côté Python,
 * sur le miroir local. C'est ce choix qui rend le filtre « posé-puis-recassé »
 * correct (le miroir a tout l'historique, pas seulement le lot courant).
 */
final class BlocksHandler implements HttpHandler {

    private static final String SQL =
        "SELECT rowid AS cp_rowid, time, user, wid, x, y, z, type, action " +
        "FROM %s WHERE rowid > ? ORDER BY rowid LIMIT ?;";

    private final ReadOnlyCoreProtectDb db;
    private final String sql;
    private final int maxPageSize;
    private final Logger log;

    BlocksHandler(ReadOnlyCoreProtectDb db, String blockTable, int maxPageSize, Logger log) {
        this.db = db;
        this.sql = String.format(SQL, blockTable);
        this.maxPageSize = maxPageSize;
        this.log = log;
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
        if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
            exchange.sendResponseHeaders(405, -1);
            exchange.close();
            return;
        }

        Map<String, String> params = Query.parse(exchange.getRequestURI().getRawQuery());
        long since = Query.longOrDefault(params.get("since"), 0L);
        int limit = (int) Query.longOrDefault(params.get("limit"), maxPageSize);
        if (limit <= 0 || limit > maxPageSize) {
            limit = maxPageSize;
        }
        final int boundLimit = limit;

        QueryResponder.respondCsvGzip(exchange, db, sql, ps -> {
            ps.setLong(1, since);
            ps.setInt(2, boundLimit);
        }, log);
    }
}
