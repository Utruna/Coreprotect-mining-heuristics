package net.utruna.xrayindexer.http;

import java.io.IOException;
import java.io.Writer;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;

/** Écriture CSV minimale et robuste (RFC 4180) pour les réponses de la passerelle. */
final class Csv {

    private Csv() {
    }

    /**
     * Écrit un ResultSet complet en CSV (ligne d'en-tête + données) dans writer.
     * Streaming : une ligne à la fois, jamais tout le résultat en mémoire.
     *
     * @return nombre de lignes de données écrites.
     */
    static long write(ResultSet rs, Writer writer) throws SQLException, IOException {
        ResultSetMetaData meta = rs.getMetaData();
        int cols = meta.getColumnCount();

        for (int i = 1; i <= cols; i++) {
            if (i > 1) writer.write(',');
            writer.write(escape(meta.getColumnLabel(i)));
        }
        writer.write('\n');

        long rows = 0;
        while (rs.next()) {
            for (int i = 1; i <= cols; i++) {
                if (i > 1) writer.write(',');
                writer.write(escape(rs.getString(i)));
            }
            writer.write('\n');
            rows++;
        }
        return rows;
    }

    static String escape(String value) {
        if (value == null) return "";
        if (value.indexOf(',') >= 0 || value.indexOf('"') >= 0
                || value.indexOf('\n') >= 0 || value.indexOf('\r') >= 0) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }
}
