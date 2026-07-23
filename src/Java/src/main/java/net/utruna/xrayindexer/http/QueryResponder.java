package net.utruna.xrayindexer.http;

import com.sun.net.httpserver.HttpExchange;
import net.utruna.xrayindexer.db.ReadOnlyCoreProtectDb;

import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.logging.Logger;
import java.util.zip.GZIPOutputStream;

/**
 * Exécute une requête en lecture seule et renvoie le résultat en CSV gzippé,
 * streamé. Chaque requête HTTP ouvre sa propre connexion (thread-safe), la
 * ferme en fin de traitement, et n'accumule jamais le résultat en mémoire.
 */
final class QueryResponder {

    /** Lie les paramètres d'un PreparedStatement (curseur, limite...). */
    interface Binder {
        void bind(PreparedStatement ps) throws SQLException;
    }

    private QueryResponder() {
    }

    static void respondCsvGzip(HttpExchange exchange, ReadOnlyCoreProtectDb db,
                               String sql, Binder binder, Logger log) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", "text/csv; charset=utf-8");
        exchange.getResponseHeaders().set("Content-Encoding", "gzip");
        // 0 => corps en chunked, adapté au streaming (taille inconnue d'avance).
        exchange.sendResponseHeaders(200, 0);

        try (Connection connection = db.open();
             PreparedStatement ps = connection.prepareStatement(sql)) {
            binder.bind(ps);
            try (ResultSet rs = ps.executeQuery();
                 OutputStream raw = exchange.getResponseBody();
                 GZIPOutputStream gz = new GZIPOutputStream(raw);
                 Writer writer = new OutputStreamWriter(gz, StandardCharsets.UTF_8)) {
                Csv.write(rs, writer);
            }
        } catch (SQLException e) {
            // En-têtes déjà envoyés : impossible de renvoyer un code d'erreur propre.
            // On coupe le flux (le client verra un gzip tronqué => erreur côté client)
            // et on trace côté serveur.
            log.warning("Erreur SQL pendant la réponse : " + e.getMessage());
        } finally {
            exchange.close();
        }
    }
}
