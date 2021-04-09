theme_paper <- function() {
    theme_minimal() +
        theme(
            legend.position = "bottom",
            axis.line = element_line()
        )
}

scale_color_algorithm <- function() {
    # Okaibe-Ito palette
    palette <- list(
        "ChenEtAl" = "#E69F00",
        "MRCoreset" = "#56B4E9",
        "Random" = "#009E73",
        "SeqCoreset" = "#F0E442",
        "StreamingCoreset" = "#0072B2"
        # "#D55E00",
        # "#CC79A7",
        # "#000000"
    )
    scale_color_manual(values = palette, aesthetics = c("fill", "color"))
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

    random <- plotdata %>% filter(algorithm == "Random")

    averages <- plotdata %>%
        mutate(algorithm_params = if_else(algorithm == "Random", "", algorithm_params)) %>%
        group_by(dataset, outliers_spec, algorithm, algorithm_params) %>%
        summarise(radius = mean(radius), total_time = mean(total_time))
    ggplot(plotdata, aes(
        x = radius, y = total_time, color = algorithm,
        tooltip = str_c(
            "time=", scales::number(total_time, accuracy = 0.01),
            " radius=", scales::number(radius, accuracy = 0.1),
            "\nparameters: ", algorithm_params
        )
    )) +
        geom_point(data = random, alpha = 0.5, size = 0.5) +
        geom_point_interactive(data = averages, size = 1.2) +
        scale_y_continuous(trans = "log10") +
        scale_color_algorithm() +
        labs(y = "total time (s)", x = "radius", title = title) +
        facet_wrap(vars(dataset, outliers_spec), scales = "free") +
        theme_paper()
}

do_plot_param_influence <- function(plotdata) {
    plotdata <- plotdata %>%
        filter(
            algorithm %in% c("MRCoreset", "StreamingCoreset", "SeqCoreset")
        ) %>%
        filter((algorithm != "MRCoreset") | (threads == 16)) %>%
        mutate(
            algorithm_params = map(
                algorithm_params,
                ~ jsonlite::fromJSON(.) %>% as.data.frame()
            )
        ) %>%
        unnest(algorithm_params) %>%
        mutate(
            tau = tau * threads,
            tau = factor(tau, ordered = T),
            xval = rank(tau)
        ) %>%
        group_by(dataset, algorithm, xval, tau, threads) %>%
        summarise(
            radius = mean(radius),
            proxy_radius = mean(proxy_radius),
            coreset_size = mean(coreset_size)
        )

    labs <- plotdata %>% select(xval, tau)
    labels <- pull(labs, tau)
    breaks <- pull(labs, xval)

    pos <- position_dodge(0.9)
    ggplot(plotdata, aes(
        x = tau,
        y = proxy_radius,
        fill = algorithm,
        color = algorithm,
        group = algorithm
    )) +
        geom_col(
            aes(y = radius),
            position = position_dodge(),
            alpha = 0.2
        ) +
        geom_point(
            aes(y = radius),
            position = pos
        ) +
        geom_point(
            aes(y = proxy_radius),
            position = pos,
            shape = 3
        ) +
        scale_color_algorithm() +
        labs(
            y = "radius",
            x = "tau",
            caption = str_wrap(
                "Dots and bars denote the radius of the final clustering
                (with outliers), whereas crosses denote the radius of
                the proxies (without outliers)"
            )
        ) +
        theme_paper()
}

do_plot_param_influence(table_result())