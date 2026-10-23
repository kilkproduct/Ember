#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ember::json {

enum class Type { Null, Bool, Number, String, Array, Object };

struct Value {
    Type type = Type::Null;
    bool b = false;
    double num = 0.0;
    std::string str;       // UTF-8 string
    std::string latin1;    // 1-codepoint-per-byte raw string for BPE tokens
    std::vector<Value> arr;
    std::map<std::string, Value> obj;

    bool has(const std::string& k) const { return obj.find(k) != obj.end(); }
    const Value& at(const std::string& k) const {
        static const Value kNull{};
        auto it = obj.find(k);
        return it == obj.end() ? kNull : it->second;
    }
    int64_t int_or(const std::string& k, int64_t def = 0) const {
        auto it = obj.find(k);
        return (it != obj.end() && it->second.type == Type::Number) ? static_cast<int64_t>(it->second.num) : def;
    }
    double double_or(const std::string& k, double def = 0.0) const {
        auto it = obj.find(k);
        return (it != obj.end() && it->second.type == Type::Number) ? it->second.num : def;
    }
    bool bool_or(const std::string& k, bool def = false) const {
        auto it = obj.find(k);
        return (it != obj.end() && it->second.type == Type::Bool) ? it->second.b : def;
    }
    std::string str_or(const std::string& k, const std::string& def = "") const {
        auto it = obj.find(k);
        return (it != obj.end() && it->second.type == Type::String) ? it->second.str : def;
    }
};

class Parser {
public:
    Parser(const char* src, size_t len) : s_(src), n_(len) {}

    Value parse_root() {
        skip_ws();
        Value v = parse_value();
        skip_ws();
        return v;
    }

private:
    const char* s_;
    size_t n_;
    size_t p_ = 0;

    void skip_ws() {
        while (p_ < n_ && (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\r' || s_[p_] == '\n')) ++p_;
    }

    char peek() const { return p_ < n_ ? s_[p_] : '\0'; }
    char get() {
        if (p_ >= n_) throw std::runtime_error("Unexpected end of JSON input");
        return s_[p_++];
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp <= 0x7F) {
            out.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    uint16_t parse_hex4() {
        if (p_ + 4 > n_) throw std::runtime_error("Truncated \\uXXXX escape");
        uint16_t val = 0;
        for (int i = 0; i < 4; ++i) {
            char c = s_[p_++];
            val <<= 4;
            if (c >= '0' && c <= '9') val |= (c - '0');
            else if (c >= 'a' && c <= 'f') val |= (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') val |= (c - 'A' + 10);
            else throw std::runtime_error("Invalid hex digit in \\uXXXX escape");
        }
        return val;
    }

    Value parse_string() {
        if (get() != '"') throw std::runtime_error("Expected '\"'");
        Value v;
        v.type = Type::String;
        while (p_ < n_) {
            unsigned char ch = static_cast<unsigned char>(get());
            if (ch == '"') return v;
            if (ch == '\\') {
                char esc = get();
                uint32_t cp = 0;
                switch (esc) {
                    case '"': cp = '"'; break;
                    case '\\': cp = '\\'; break;
                    case '/': cp = '/'; break;
                    case 'b': cp = '\b'; break;
                    case 'f': cp = '\f'; break;
                    case 'n': cp = '\n'; break;
                    case 'r': cp = '\r'; break;
                    case 't': cp = '\t'; break;
                    case 'u': {
                        uint16_t hi = parse_hex4();
                        if (hi >= 0xD800 && hi <= 0xDBFF && p_ + 2 < n_ && s_[p_] == '\\' && s_[p_ + 1] == 'u') {
                            p_ += 2;
                            uint16_t lo = parse_hex4();
                            cp = 0x10000U + ((static_cast<uint32_t>(hi - 0xD800) << 10) | (lo - 0xDC00));
                        } else {
                            cp = hi;
                        }
                        break;
                    }
                    default: throw std::runtime_error("Invalid JSON string escape");
                }
                append_utf8(v.str, cp);
                v.latin1.push_back(static_cast<char>(static_cast<uint8_t>(cp & 0xFF)));
            } else {
                v.str.push_back(static_cast<char>(ch));
                v.latin1.push_back(static_cast<char>(ch));
            }
        }
        throw std::runtime_error("Unterminated JSON string");
    }

    Value parse_value() {
        skip_ws();
        char c = peek();
        if (c == '"') return parse_string();
        if (c == '{') {
            get();
            Value v;
            v.type = Type::Object;
            skip_ws();
            if (peek() == '}') { get(); return v; }
            while (true) {
                skip_ws();
                Value key = parse_string();
                skip_ws();
                if (get() != ':') throw std::runtime_error("Expected ':' in object");
                v.obj.emplace(std::move(key.str), parse_value());
                skip_ws();
                char sep = get();
                if (sep == '}') break;
                if (sep != ',') throw std::runtime_error("Expected ',' or '}' in object");
            }
            return v;
        }
        if (c == '[') {
            get();
            Value v;
            v.type = Type::Array;
            skip_ws();
            if (peek() == ']') { get(); return v; }
            while (true) {
                v.arr.push_back(parse_value());
                skip_ws();
                char sep = get();
                if (sep == ']') break;
                if (sep != ',') throw std::runtime_error("Expected ',' or ']' in array");
            }
            return v;
        }
        if (c == 't') {
            if (p_ + 4 > n_ || std::string(s_ + p_, 4) != "true") throw std::runtime_error("Invalid literal");
            p_ += 4;
            Value v; v.type = Type::Bool; v.b = true; return v;
        }
        if (c == 'f') {
            if (p_ + 5 > n_ || std::string(s_ + p_, 5) != "false") throw std::runtime_error("Invalid literal");
            p_ += 5;
            Value v; v.type = Type::Bool; v.b = false; return v;
        }
        if (c == 'n') {
            if (p_ + 4 > n_ || std::string(s_ + p_, 4) != "null") throw std::runtime_error("Invalid literal");
            p_ += 4;
            return Value{};
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            char* end = nullptr;
            double val = std::strtod(s_ + p_, &end);
            if (end == s_ + p_) throw std::runtime_error("Invalid JSON number");
            p_ = static_cast<size_t>(end - s_);
            Value v; v.type = Type::Number; v.num = val; return v;
        }
        throw std::runtime_error("Unexpected character in JSON");
    }
};

inline Value parse(const std::string& text) {
    return Parser(text.data(), text.size()).parse_root();
}

}  // namespace ember::json
