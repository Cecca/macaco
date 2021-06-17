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

table_result <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), here("kcmkc-results.sqlite"))
    results <- tbl(conn, "result") %>%
        collect() %>%
        replace_na(list(threads = 1)) %>%
        # filter(algorithm != "MRCoreset") %>%
        filter(dataset %in% c(
            "Random",
            "MusixMatch",
            "MusixMatch-sample-10000",
            "Wikipedia-sample-10000",
            # "Wikipedia-euclidean-sample-10000",
            "Wikipedia"
            # "Wikipedia-euclidean"
        )) %>%
        rowwise() %>%
        mutate(
            dimensions = access_json(dataset_params, "dimensions") %>% as.numeric(),
            dimensions = if_else(str_detect(dataset, "MusixMatch"), 5000, dimensions)
        ) %>% 
        mutate(
            distance = if_else(str_detect(dataset, "euclidean"), "euclidean", "cosine"),
            is_sample = str_detect(dataset, "sample"),
            rank = matroid_rank(constraint_params),
            tau = access_json(algorithm_params, "tau"),
            workers = threads * str_count(hosts, ":"),
            workers = if_else(is.na(workers), 1, workers)
        ) %>%
        unnest(rank) %>%
        # filter(outliers_spec %in% c("Percentage(0.01)")) %>%
        filter(dimensions %in% c(5000, 10, 3)) %>%
        filter(!(str_detect(dataset, "Wikipedia") & (rank == 100))) %>%
        mutate(
            total_time = set_units(total_time_ms, "ms"),
            coreset_time = set_units(coreset_time_ms, "ms"),
            solution_time = set_units(solution_time_ms, "ms")
        ) %>%
        # filter(algorithm == "ChenEtAl") %>% distinct(dataset, dimensions) %>% print()
        select(-ends_with("_ms"))

    best <- results %>%
        group_by(dataset, rank) %>%
        summarise(best_radius = min(radius))

    results <- inner_join(results, best) %>%
        mutate(ratio_to_best = radius / best_radius)
    
    DBI::dbDisconnect(conn)
    results
}