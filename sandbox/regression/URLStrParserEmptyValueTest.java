package org.apache.dubbo.common;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class URLStrParserEmptyValueTest {
    @Test
    void decodedEmptyParameterValue() {
        assertEquals("", URLStrParser.parseDecodedStr("dubbo://127.0.0.1?key_name=").getParameter("key_name"));
    }

    @Test
    void encodedEmptyParameterValue() {
        String input = URL.encode("dubbo://127.0.0.1?key_name=");
        assertEquals("", URLStrParser.parseEncodedStr(input).getParameter("key_name"));
    }
}
