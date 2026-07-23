package net.utruna.xrayindexer.http;

import com.sun.net.httpserver.HttpContext;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import net.utruna.xrayindexer.db.ReadOnlyCoreProtectDb;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Logger;

/**
 * Serveur HTTP intégré (com.sun.net.httpserver, fourni par le JDK — aucune
 * dépendance ajoutée) qui expose la base CoreProtect en lecture seule :
 *
 *   GET /health              -> { status, max_rowid }
 *   GET /blocks?since=&limit= -> tranche de co_block (CSV gzip, paginée par rowid)
 *   GET /users               -> co_user      (id, uuid, user)      CSV gzip
 *   GET /materials           -> co_material_map (id, material)     CSV gzip
 *   GET /worlds              -> co_world      (id, world)          CSV gzip
 *
 * Toutes les routes exigent un jeton Bearer. Le serveur tourne sur son propre
 * pool de threads, indépendant du thread principal du serveur Minecraft.
 */
public final class GatewayServer {

    private final HttpServer httpServer;
    private final ExecutorService executor;
    private final Logger log;

    public GatewayServer(String bind, int port, int backlog, String token, int threads,
                         ReadOnlyCoreProtectDb db, String blockTable, String userTable,
                         String materialTable, String worldTable, int maxPageSize, Logger log)
            throws IOException {
        this.log = log;
        this.httpServer = HttpServer.create(new InetSocketAddress(bind, port), backlog);
        this.executor = Executors.newFixedThreadPool(Math.max(1, threads));
        this.httpServer.setExecutor(executor);

        AuthFilter auth = new AuthFilter(token);

        register("/health", new HealthHandler(db, blockTable), auth);
        register("/blocks", new BlocksHandler(db, blockTable, maxPageSize, log), auth);
        register("/users",
                new MapHandler(db, "SELECT id, uuid, user FROM " + userTable + ";", log), auth);
        register("/materials",
                new MapHandler(db, "SELECT id, material FROM " + materialTable + ";", log), auth);
        register("/worlds",
                new MapHandler(db, "SELECT id, world FROM " + worldTable + ";", log), auth);
    }

    private void register(String path, HttpHandler handler, AuthFilter auth) {
        HttpContext ctx = httpServer.createContext(path, handler);
        ctx.getFilters().addAll(List.of(auth));
    }

    public void start() {
        httpServer.start();
        log.info("Passerelle XRay démarrée sur " + httpServer.getAddress());
    }

    public void stop() {
        httpServer.stop(2); // laisse jusqu'à 2 s aux requêtes en cours
        executor.shutdownNow();
        log.info("Passerelle XRay arrêtée.");
    }
}
