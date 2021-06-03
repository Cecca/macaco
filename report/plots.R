theme_paper <- function() {
    theme_minimal() +
        theme(
            legend.position = "bottom",
            axis.line = element_line()
        )
}

algopalette <- list(
    "ChenEtAl" = "#E69F00",
    "MRCoreset" = "#56B4E9",
    "Random" = "#009E73",
    "SeqCoreset" = "#F0E442",
    "StreamingCoreset" = "#0072B2"
    # "#D55E00",
    # "#CC79A7",
    # "#000000"
)

algopalette <- list(
    "ChenEtAl" = "#4e79a7",
    "MRCoreset" = "#59a14f",
    "SeqCoreset" = "#f28e2b",
    "StreamingCoreset" = "#edc948",
    "Random" = "#76b7b2"
)

scale_color_algorithm <- function() {
    scale_color_manual(values = algopalette, aesthetics = c("fill", "color"))
}

do_plot_tradeoff <- function(data) {
    plotdata <- data %>%
        mutate(total_time = set_units(total_time, "s") %>% drop_units()) %>%
        mutate(dataset = fct_reorder2(dataset, is_sample, distance))

    random <- plotdata %>% 
        filter(algorithm == "Random")

    random_radii <- random %>%
        group_by(dataset, rank, outliers_spec) %>%
        summarise(
            random_radius = mean(radius),
            random_ratio_to_best = mean(ratio_to_best)
        )

    averages <- plotdata %>%
        mutate(algorithm_params = if_else(algorithm == "Random", "", algorithm_params)) %>%
        group_by(dataset, rank, workers, outliers_spec, algorithm, algorithm_params) %>%
        summarise(radius = mean(radius), ratio_to_best = mean(ratio_to_best), total_time = mean(total_time), coreset_size = mean(coreset_size)) %>%
        ungroup()

    draw <- function(data) {
        data %>% distinct(dataset, rank) %>% print()
        title <- data %>% distinct(dataset) %>% pull()
        random <- semi_join(random, data)
        random_radii <- semi_join(random_radii, data)
        ggplot(data, aes(
            x = ratio_to_best, y = total_time, color = algorithm,
            tooltip = str_c(
                "time=", scales::number(total_time, accuracy = 0.01),
                " radius=", scales::number(radius, accuracy = 0.001),
                " ratio to best=", scales::number(ratio_to_best, accuracy = 0.001),
                "\ncoreset size=", if_else(
                    algorithm %in% c("SeqCoreset", "StreamingCoreset", "MRCoreset"),
                    scales::number(coreset_size),
                    "-"
                ),
                "\nparameters: ", if_else(
                    algorithm == "MRCoreset",
                    str_c(algorithm_params, " ", workers, " workers"),
                    algorithm_params
                )
            )
        )) +
            geom_vline(
                aes(xintercept=random_ratio_to_best),
                data = random_radii,
                color = algopalette["Random"]
            ) +
            # geom_point(data = random, alpha = 0.5, size = 0.5) +
            geom_point_interactive(size = 1.2, stat="summary", fun.data=mean_se) +
            # scale_y_continuous(trans = "log10") +
            scale_color_algorithm() +
            labs(y = "total time (s)", x = "radius", title=title) +
            # facet_grid(vars(dataset), vars(rank), scales = "free") +
            facet_wrap(vars(rank), scales = "free") +
            theme_paper()
    }

    p_musixmatch <- averages %>% filter(dataset == "MusixMatch") %>% draw()
    p_musixmatch_sample <- averages %>% filter(dataset == "MusixMatch-sample-10000") %>% draw()
    p_wiki <- averages %>% filter(dataset == "Wikipedia") %>% draw()
    p_wiki_sample <- averages %>% filter(dataset == "Wikipedia-sample-10000") %>% draw()
    p_random <- averages %>% filter(dataset == "Random") %>% draw()
    (p_musixmatch /p_musixmatch_sample / p_wiki / p_wiki_sample / p_random / guide_area()) +
        plot_layout(guides = 'collect')
}

do_plot_time <- function(data, coreset_only=F) {
    coresets <- filter(data, str_detect(algorithm, "Coreset")) %>%
        rowwise() %>%
        mutate(tau = access_json(algorithm_params, "tau")) %>%
        ungroup() %>%
        mutate(algorithm = if_else(
            algorithm == "MRCoreset",
            str_c(algorithm, "-", workers),
            algorithm
        )) %>%
        mutate(
            algorithm = factor(algorithm, ordered = T, levels = c(
                "SeqCoreset",
                "StreamingCoreset",
                "MRCoreset-1",
                "MRCoreset-2",
                "MRCoreset-4",
                "MRCoreset-8",
                "MRCoreset-16"
            ))
        ) %>%
        group_by(dataset, rank, algorithm, tau) %>%
        summarise(solution_time=mean(solution_time), coreset_time=mean(coreset_time))

    times <- select(
            coresets, rank, algorithm, tau, dataset, 
            solution=solution_time, coreset=coreset_time
        ) %>%
        pivot_longer(solution:coreset, names_to="component", values_to="time") %>%
        mutate(time = set_units(time, "s") %>% drop_units())
    assertthat::assert_that(2*nrow(coresets) == nrow(times))

    if (coreset_only) {
        times <- filter(times, component == "coreset")
    }

    times %>%
    ggplot(aes(x=algorithm, y=time, fill=component)) +
        geom_bar_interactive(aes(
            tooltip = str_c(
                "*", component, "*\n",
                "time=", scales::number(time, accuracy = 0.01)
            )
        ), stat="summary", fun.data=mean_se) +
        facet_grid(vars(tau), vars(dataset, rank), scales="free") +
        coord_flip() +
        theme_paper()
}


do_plot_samples <- function(data) {
    data %>% 
        filter(algorithm != "Random") %>%
        filter(outliers_spec == "Percentage(0.01)") %>%
        filter(dataset %in% c("MusixMatch-sample-10000", "Wikipedia-sample-10000", "Random")) %>%
        filter(rank %in% c(20, 10)) %>%
        group_by(dataset, algorithm, rank, dataset_params, algorithm_params, tau, workers) %>%
        summarise(
            total_time = mean(total_time),
            ratio_to_best = mean(ratio_to_best)
        ) %>%
        mutate(flabel = str_c(dataset, " (rank ", rank, ")")) %>%
        ggplot(aes(ratio_to_best, total_time, color=algorithm)) +
        geom_point() +
        geom_text_repel(aes(label = tau), show.legend=F) +
        scale_y_unit(trans="log10", labels=scales::number_format()) +
        scale_color_algorithm() +
        facet_wrap(vars(flabel), scales="free") +
        theme_paper()
}
