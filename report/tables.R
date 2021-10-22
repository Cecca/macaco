matroid_rank <- function(constraint_params) {
    purrr::map(
        constraint_params,
        function(p) {
            if (str_detect(p, "transversal")) {
                length(jsonlite::fromJSON(p)$transversal$topics)
            } else if (str_detect(p, "partition")) {
                categories <- jsonlite::fromJSON(p)$partition$categories
                sum(purrr::as_vector(categories))
            } else {
                stop("unknown matroid type")
            }
        }
    )
}

access_json <- function(jstr, name) {
    if (str_length(jstr) == 0) {
        return(NA)
    }
    v <- jsonlite::fromJSON(jstr)[[name]]
    if (is.null(v)) {
        NA
    } else {
        v
    }
}

table_dataset <- function(dataset_names) {
    conn <- DBI::dbConnect(RSQLite::SQLite(), here("kcmkc-results.sqlite"))
    results <- tbl(conn, "result") %>%
        filter(dataset %in% c(dataset_names)) %>%
        collect()
    DBI::dbDisconnect(conn)
    results
}

table_result <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), here("kcmkc-results.sqlite"))
    results <- tbl(conn, "result") %>%
        collect() %>%
        replace_na(list(threads = 1)) %>%
        rowwise() %>%
        mutate(
            dimensions = access_json(dataset_params, "dimensions") %>% as.numeric(),
            dimensions = if_else(str_detect(dataset, "MusixMatch"), 5000, dimensions),
            dimensions = if_else(str_detect(dataset, "Higgs"), 7, dimensions),
            dimensions = if_else(str_detect(dataset, "Phones"), 3, dimensions)
        ) %>% 
        mutate(
            wiki_topics = access_json(dataset_params, "topics"),
            is_sample = str_detect(dataset, "sample"),
            rank = matroid_rank(constraint_params),
            tau = access_json(algorithm_params, "tau"),
            workers = threads * str_count(hosts, ":"),
            workers = if_else(is.na(workers), 1, workers)
        ) %>%
        unnest(rank) %>%
        mutate(
            total_time = set_units(total_time_ms, "ms"),
            coreset_time = set_units(coreset_time_ms, "ms"),
            solution_time = set_units(solution_time_ms, "ms")
        ) %>%
        filter((is.na(wiki_topics)) | ((wiki_topics == rank) & (wiki_topics != 10))) %>%
        filter((dataset != "MusixMatch") | (rank != 20)) %>%
        select(-ends_with("_ms")) %>%
        mutate(outliers_spec = str_replace(outliers_spec, "Percentage", "P")) %>%
        mutate(outliers_spec = str_replace(outliers_spec, "Fixed", "F"))

    best <- results %>%
        group_by(dataset, outliers_spec, rank) %>%
        summarise(best_radius = min(radius))

    results <- inner_join(results, best) %>%
        mutate(ratio_to_best = radius / best_radius)
    
    DBI::dbDisconnect(conn)
    results
}