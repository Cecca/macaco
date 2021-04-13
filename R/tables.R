matroid_rank <- function(constraint_params) {
    purrr::map(
        constraint_params,
        function(p) {
            if (str_detect(p, "transversal")) {
                length(jsonlite::fromJSON(p)$transversal$topics)
            } else {
                stop()
            }
        }
    )
}

table_result <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), here("kcmkc-results.sqlite"))
    results <- tbl(conn, "result") %>%
        collect() %>%
        replace_na(list(threads = 1)) %>%
        filter(dataset %in% c(
            "Wikipedia-sample-10000",
            # "Wikipedia-euclidean-sample-10000",
            "Wikipedia"
            # "Wikipedia-euclidean"
        )) %>%
        mutate(
            distance = if_else(str_detect(dataset, "euclidean"), "euclidean", "cosine"),
            is_sample = str_detect(dataset, "sample"),
            rank = matroid_rank(constraint_params)
        ) %>%
        unnest(rank) %>%
        filter(outliers_spec %in% c("Percentage(0.01)")) %>%
        mutate(
            total_time = set_units(total_time_ms, "ms"),
            coreset_time = set_units(coreset_time_ms, "ms"),
            solution_time = set_units(solution_time_ms, "ms")
        ) %>%
        select(-ends_with("_ms"))
    DBI::dbDisconnect(conn)
    results
}