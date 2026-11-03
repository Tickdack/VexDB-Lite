#include "duckdb.hpp"
#include "core_functions_extension.hpp"
#include "duckdb/main/materialized_query_result.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace duckdb;

static void ExecOrThrow(Connection &con, const std::string &sql) {
    auto result = con.Query(sql);
    if (!result) {
        throw std::runtime_error("query returned null result: " + sql);
    }
    if (result->HasError()) {
        result->ThrowError("query failed: " + sql + "\n");
    }
}

static std::string QueryCol0(Connection &con, const std::string &sql) {
    auto result = con.Query(sql);
    if (!result) {
        throw std::runtime_error("query returned null result: " + sql);
    }
    if (result->HasError()) {
        result->ThrowError("query failed: " + sql + "\n");
    }
    auto &mat = result->Cast<MaterializedQueryResult>();
    std::string out;
    for (idx_t row = 0; row < mat.RowCount(); row++) {
        if (row) {
            out += ",";
        }
        out += mat.GetValue(0, row).ToString();
    }
    return out;
}

static std::string QueryAll(Connection &con, const std::string &sql) {
    auto result = con.Query(sql);
    if (!result) {
        throw std::runtime_error("query returned null result: " + sql);
    }
    if (result->HasError()) {
        result->ThrowError("query failed: " + sql + "\n");
    }
    auto &mat = result->Cast<MaterializedQueryResult>();
    std::string out;
    for (idx_t row = 0; row < mat.RowCount(); row++) {
        for (idx_t col = 0; col < mat.ColumnCount(); col++) {
            out += mat.GetValue(col, row).ToString();
            out += '\n';
        }
    }
    return out;
}

static void ExpectIds(Connection &con, const std::string &sql, const std::string &expected) {
        auto got = QueryCol0(con, sql);
    if (got != expected) {
        throw std::runtime_error("expected [" + expected + "] from:\n" + sql + "\ngot [" + got + "]");
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: smoke_filtered_search <extension_path>\n";
        return 2;
    }

    try {
        DBConfig config;
        config.SetOptionByName("allow_unsigned_extensions", true);
        DuckDB db(nullptr, &config);
        db.LoadStaticExtension<CoreFunctionsExtension>();
        Connection con(db);

        ExecOrThrow(con, "LOAD '" + std::string(argv[1]) + "'");
        ExecOrThrow(con, "SET vexdb_brute_force_threshold=0");
        ExecOrThrow(con, "SET vexdb_ef_search=40");
        ExecOrThrow(con, "CREATE TABLE items (id INTEGER, vec FLOAT[2], category VARCHAR, year INTEGER, tag VARCHAR)");
        ExecOrThrow(con,
                    "INSERT INTO items SELECT i, "
                    "[i::FLOAT, 0.0::FLOAT]::FLOAT[2], "
                    "CASE WHEN i % 2 = 0 THEN 'even' ELSE 'odd' END, "
                    "2000 + (i % 5), "
                    "CASE WHEN i > 150 THEN 'keep' ELSE 'drop' END "
                    "FROM range(1, 201) t(i)");
        ExecOrThrow(con, "CREATE INDEX idx_items ON items USING GRAPH_INDEX (vec)");

        auto explain = QueryAll(
            con, "EXPLAIN SELECT id FROM items WHERE tag = 'keep' "
                 "ORDER BY l2_distance(vec, [0.0, 0.0]::FLOAT[2]) LIMIT 5");
        if (explain.find("VEXDB_INDEX_SCAN") == std::string::npos) {
            throw std::runtime_error("expected VEXDB_INDEX_SCAN in filtered plan, got:\n" + explain);
        }
        if (explain.find("in-filter") == std::string::npos && explain.find("Filter") == std::string::npos) {
            // ParamsToString may render as Algorithm=in-filter or Filter=pushed
            std::cerr << "warn: plan missing explicit filter label:\n" << explain << "\n";
        }

        // Killer case: nearest neighbours of [0,0] are ids 1,2,3,... none of the
        // first 150 have tag='keep'. Post-filter returns empty; in-filter must
        // return 151..155.
        ExpectIds(con,
                  "SELECT id FROM items WHERE tag = 'keep' "
                  "ORDER BY l2_distance(vec, [0.0, 0.0]::FLOAT[2]) LIMIT 5",
                  "151,152,153,154,155");

        ExpectIds(con,
                  "SELECT id FROM items WHERE category = 'even' "
                  "ORDER BY l2_distance(vec, [0.0, 0.0]::FLOAT[2]) LIMIT 10",
                  "2,4,6,8,10,12,14,16,18,20");

        ExpectIds(con,
                  "SELECT id FROM items WHERE category = 'even' AND year = 2003 "
                  "ORDER BY l2_distance(vec, [0.0, 0.0]::FLOAT[2]) LIMIT 3",
                  "8,18,28");

        ExpectIds(con,
                  "SELECT id FROM items WHERE category = 'odd' AND year = 2001 "
                  "ORDER BY l2_distance(vec, [0.0, 0.0]::FLOAT[2]) LIMIT 10",
                  "1,11,21,31,41,51,61,71,81,91");

        auto empty = QueryCol0(con,
                                   "SELECT id FROM items WHERE category = 'nope' "
                                   "ORDER BY l2_distance(vec, [0.0, 0.0]::FLOAT[2]) LIMIT 5");
        if (!empty.empty()) {
            throw std::runtime_error("expected empty result for category='nope', got [" + empty + "]");
        }

        ExpectIds(con,
                  "SELECT id FROM items "
                  "ORDER BY l2_distance(vec, [0.0, 0.0]::FLOAT[2]) LIMIT 3",
                  "1,2,3");

        std::cout << "smoke_filtered_search: ok\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "smoke_filtered_search: fail: " << ex.what() << "\n";
        return 1;
    }
}
