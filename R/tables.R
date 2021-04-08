table_result <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), here("kcmkc-results.sqlite"))
    results <- tbl(conn, "result") %>%
        collect() %>%
        replace_na(list(threads = 1)) %>%
        filter(dataset %in% c(
            "Wikipedia-sample-50000",
            "Wikipedia-euclidean-sample-50000",
            "Wikipedia",
            "Wikipedia-euclidean"
        )) %>%
        mutate(
            distance = if_else(str_detect(dataset, "euclidean"), "euclidean", "cosine"),
            is_sample = str_detect(dataset, "sample")
        ) %>%
        filter(outliers_spec %in% c("Percentage(0.01)")) %>%
        mutate(total_time = set_units(total_time_ms, "ms")) %>%
        select(-total_time_ms)
    DBI::dbDisconnect(conn)
    results
}