package net.utruna.xrayindexer.db;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ReadOnlyCoreProtectDbTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void opensExistingDatabaseForReadsAndRejectsWrites() throws Exception {
        Path databasePath = temporaryDirectory.resolve("coreprotect.db");
        try (Connection connection = DriverManager.getConnection("jdbc:sqlite:" + databasePath);
             Statement statement = connection.createStatement()) {
            statement.execute("CREATE TABLE sample (id INTEGER)");
            statement.execute("INSERT INTO sample VALUES (7)");
        }

        ReadOnlyCoreProtectDb database = new ReadOnlyCoreProtectDb(databasePath);
        try (Connection connection = database.open();
             Statement statement = connection.createStatement();
             var resultSet = statement.executeQuery("SELECT id FROM sample")) {
            resultSet.next();
            assertEquals(7, resultSet.getInt(1));
            assertThrows(SQLException.class,
                    () -> statement.execute("INSERT INTO sample VALUES (8)"));
        }
        assertEquals(databasePath, database.path());
    }

    @Test
    void rejectsMissingDatabase() {
        Path missingPath = temporaryDirectory.resolve("missing.db");

        assertThrows(IOException.class, () -> new ReadOnlyCoreProtectDb(missingPath));
        assertEquals(false, Files.exists(missingPath));
    }
}