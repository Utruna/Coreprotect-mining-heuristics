package net.utruna.xrayindexer.http;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.StringWriter;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;

import org.junit.jupiter.api.Test;

class CsvTest {

    @Test
    void escapeQuotesFieldsThatContainCsvSpecialCharacters() {
        assertEquals("", Csv.escape(null));
        assertEquals("plain", Csv.escape("plain"));
        assertEquals("\"a,b\"", Csv.escape("a,b"));
        assertEquals("\"a\"\"b\"", Csv.escape("a\"b"));
        assertEquals("\"first\nsecond\"", Csv.escape("first\nsecond"));
    }

    @Test
    void writeOutputsHeaderRowsAndEscapedValues() throws Exception {
        try (Connection connection = DriverManager.getConnection("jdbc:sqlite::memory:");
             Statement statement = connection.createStatement()) {
            statement.execute("CREATE TABLE records (id INTEGER, label TEXT)");
            statement.execute("INSERT INTO records VALUES (1, 'plain')");
            statement.execute("INSERT INTO records VALUES (2, 'a,b')");

            try (ResultSet resultSet = statement.executeQuery(
                    "SELECT id AS record_id, label FROM records ORDER BY id")) {
                StringWriter writer = new StringWriter();

                assertEquals(2L, Csv.write(resultSet, writer));
                assertEquals("record_id,label\n1,plain\n2,\"a,b\"\n", writer.toString());
            }
        }
    }
}