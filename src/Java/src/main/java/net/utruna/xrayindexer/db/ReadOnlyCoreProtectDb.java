package net.utruna.xrayindexer.db;

import org.sqlite.SQLiteConfig;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Properties;

/**
 * Ouvre le fichier database.db de CoreProtect en lecture seule STRICTE.
 *
 * Deux garde-fous indépendants, volontairement redondants :
 *   1. SQLiteConfig.setReadOnly(true) -> le driver JDBC refuse toute écriture
 *      au niveau connexion (SQLITE_OPEN_READONLY côté natif).
 *   2. PRAGMA query_only = ON -> même si un bug introduisait un jour un
 *      INSERT/UPDATE quelque part dans le code, SQLite le refuse lui-même.
 *
 * Cette classe ne crée, ne modifie, ne supprime jamais rien dans le fichier
 * qu'elle ouvre. Toute tentative d'écriture doit lever une SQLException.
 *
 * Elle n'expose pas UNE connexion partagée : le serveur HTTP sert plusieurs
 * requêtes en parallèle, et une Connection SQLite n'est pas conçue pour un usage
 * concurrent. On ouvre donc une connexion neuve par requête (open()) — c'est bon
 * marché sur un fichier SQLite en WAL, et chaque requête referme la sienne.
 */
public final class ReadOnlyCoreProtectDb {

    private final Path dbPath;
    private final Properties roProperties;

    public ReadOnlyCoreProtectDb(Path dbPath) throws SQLException, IOException {
        if (!Files.exists(dbPath)) {
            throw new IOException("Base CoreProtect introuvable : " + dbPath.toAbsolutePath());
        }
        this.dbPath = dbPath;

        SQLiteConfig config = new SQLiteConfig();
        config.setReadOnly(true);
        this.roProperties = config.toProperties();

        verifyStrictlyReadOnly();
    }

    /**
     * Ouvre une connexion neuve en lecture seule. À fermer par l'appelant
     * (try-with-resources). PRAGMA query_only en second garde-fou, busy_timeout
     * pour ne pas échouer instantanément si CoreProtect tient brièvement un lock.
     */
    public Connection open() throws SQLException {
        String url = "jdbc:sqlite:" + dbPath.toAbsolutePath();
        Connection connection = DriverManager.getConnection(url, roProperties);
        try (Statement st = connection.createStatement()) {
            st.execute("PRAGMA query_only = ON;");
            st.execute("PRAGMA busy_timeout = 5000;");
        }
        return connection;
    }

    /**
     * Vérification active au démarrage : si les deux garde-fous échouaient
     * silencieusement (version de driver, config système...), on préfère planter
     * au démarrage plutôt que de découvrir le problème en modifiant la base
     * CoreProtect en prod.
     */
    private void verifyStrictlyReadOnly() throws SQLException {
        try (Connection connection = open();
             Statement st = connection.createStatement()) {
            st.execute("CREATE TABLE __xraygateway_should_never_be_created__ (x INTEGER);");
        } catch (SQLException expected) {
            return; // Comportement attendu : l'écriture doit échouer.
        }
        throw new SQLException(
            "ALERTE : la connexion à " + dbPath.toAbsolutePath() +
            " n'est PAS en lecture seule malgré la configuration. " +
            "Le plugin refuse de continuer pour ne pas risquer d'écrire dans CoreProtect."
        );
    }

    public Path path() {
        return dbPath;
    }
}
