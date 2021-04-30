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

scale_color_algorithm <- function() {
    # Okaibe-Ito palette
    scale_color_manual(values = algopalette, aesthetics = c("fill", "color"))
}

do_plot_tradeoff <- function(data) {
    assertthat::assert_that(
        count(distinct(data, rank)) == 1,
        msg = str_c(
            "Should have a single rank in do_plot_tradeoff: ",
            pull(distinct(data, rank))
        )
    )

    title <- str_c("Rank ", distinct(data, rank) %>% pull())

    plotdata <- data %>%
        mutate(total_time = set_units(total_time, "s") %>% drop_units()) %>%
        mutate(dataset = fct_reorder2(dataset, is_sample, distance))

    random <- plotdata %>% 
        filter(algorithm == "Random")

    random_radii <- random %>%
        group_by(dataset, outliers_spec) %>%
        summarise(random_radius = mean(radius))

    averages <- plotdata %>%
        mutate(algorithm_params = if_else(algorithm == "Random", "", algorithm_params)) %>%
        group_by(dataset, workers, outliers_spec, algorithm, algorithm_params) %>%
        summarise(radius = mean(radius), total_time = mean(total_time), coreset_size = mean(coreset_size))
    ggplot(plotdata, aes(
        x = radius, y = total_time, color = algorithm,
        tooltip = str_c(
            "time=", scales::number(total_time, accuracy = 0.01),
            " radius=", scales::number(radius, accuracy = 0.001),
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
            aes(xintercept=random_radius),
            data = random_radii,
            color = algopalette["Random"]
        ) +
        geom_point(data = random, alpha = 0.5, size = 0.5) +
        geom_point_interactive(data = averages, size = 1.2) +
        scale_y_continuous(trans = "log10") +
        scale_color_algorithm() +
        labs(y = "total time (s)", x = "radius", title = title) +
        facet_wrap(vars(dataset, outliers_spec), scales = "free") +
        theme_paper()
}

do_plot_time <- function(data, coreset_only=F) {
    assertthat::assert_that(
        count(distinct(data, rank)) == 1,
        msg = str_c(
            "Should have a single rank in do_plot_tradeoff: ",
            pull(distinct(data, rank))
        )
    )

    title <- str_c("Rank ", distinct(data, rank) %>% pull())

    coresets <- filter(data, str_detect(algorithm, "Coreset")) %>%
        mutate(
            algorithm_params = map(
                algorithm_params,
                ~ jsonlite::fromJSON(.) %>% as.data.frame()
            )
        ) %>%
        unnest(algorithm_params) %>%
        mutate(final_tau = tau * workers) %>%
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
        )

    times <- select(coresets, rank, algorithm, tau, dataset, outliers_spec, solution=solution_time, coreset=coreset_time) %>%
        pivot_longer(solution:coreset, names_to="component", values_to="time") %>%
        mutate(time = set_units(time, "s") %>% drop_units())

    if (coreset_only) {
        times <- filter(times, component == "coreset")
    }

    sizes <- coresets %>%
        group_by(dataset, rank, algorithm, tau, outliers_spec) %>%
        summarise(
            coreset_size = mean(coreset_size)
        )
    
    inner_join(times, sizes) %>%
    ggplot(aes(x=algorithm, y=time, fill=component)) +
        geom_bar_interactive(aes(
            tooltip = str_c(
                "*", component, "*\n",
                "time=", scales::number(time, accuracy = 0.01),
                "\ncoreset size=", if_else(
                    is.na(coreset_size),
                    "-",
                    scales::number(coreset_size)
                )
            )
        ), stat="summary") +
        facet_grid(vars(tau), vars(dataset, outliers_spec), scales="free") +
        labs(title = title) +
        coord_flip() +
        theme_paper()
}


