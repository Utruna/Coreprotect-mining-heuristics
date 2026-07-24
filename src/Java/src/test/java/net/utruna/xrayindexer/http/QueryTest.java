package net.utruna.xrayindexer.http;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Map;

import org.junit.jupiter.api.Test;

class QueryTest {

    @Test
    void parseDecodesValuesAndAcceptsValuelessParameters() {
        Map<String, String> params = Query.parse("since=42&player=Alex%20Smith&verbose");

        assertEquals("42", params.get("since"));
        assertEquals("Alex Smith", params.get("player"));
        assertEquals("", params.get("verbose"));
    }

    @Test
    void parseReturnsEmptyMapForMissingQuery() {
        assertEquals(Map.of(), Query.parse(null));
        assertEquals(Map.of(), Query.parse(""));
    }

    @Test
    void longOrDefaultUsesFallbackForMissingOrInvalidValues() {
        assertEquals(50L, Query.longOrDefault(null, 50L));
        assertEquals(50L, Query.longOrDefault("", 50L));
        assertEquals(50L, Query.longOrDefault("not-a-number", 50L));
    }

    @Test
    void longOrDefaultParsesTrimmedLongValues() {
        assertEquals(-12L, Query.longOrDefault(" -12 ", 50L));
    }
}