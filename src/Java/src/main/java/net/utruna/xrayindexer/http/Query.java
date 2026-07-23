package net.utruna.xrayindexer.http;

import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

/** Parsing minimal d'une query string HTTP (since=...&limit=...). */
final class Query {

    private Query() {
    }

    static Map<String, String> parse(String rawQuery) {
        Map<String, String> out = new HashMap<>();
        if (rawQuery == null || rawQuery.isEmpty()) {
            return out;
        }
        for (String pair : rawQuery.split("&")) {
            int eq = pair.indexOf('=');
            if (eq < 0) {
                out.put(decode(pair), "");
            } else {
                out.put(decode(pair.substring(0, eq)), decode(pair.substring(eq + 1)));
            }
        }
        return out;
    }

    static long longOrDefault(String value, long fallback) {
        if (value == null || value.isEmpty()) {
            return fallback;
        }
        try {
            return Long.parseLong(value.trim());
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    private static String decode(String s) {
        return URLDecoder.decode(s, StandardCharsets.UTF_8);
    }
}
