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
        group_by(dataset, threads, outliers_spec, algorithm, algorithm_params) %>%
        summarise(radius = mean(radius), total_time = mean(total_time))
        # inner_join(random_radii)  %>%
        # filter(radius <= 1.3 * random_radius)
    ggplot(plotdata, aes(
        x = radius, y = total_time, color = algorithm,
        tooltip = str_c(
            "time=", scales::number(total_time, accuracy = 0.01),
            " radius=", scales::number(radius, accuracy = 0.1),
            "\nparameters: ", if_else(
                algorithm == "MRCoreset",
                str_c(algorithm_params, " ", threads, " threads"),
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
        mutate(final_tau = tau * threads) %>%
        mutate(algorithm = if_else(
            algorithm == "MRCoreset",
            str_c(algorithm, "-", threads),
            algorithm
        )) %>%
        mutate(
            algorithm = factor(algorithm, ordered = T, levels = c(
                "SeqCoreset",
                "StreamingCoreset",
                "MRCoreset-2",
                "MRCoreset-4",
                "MRCoreset-8",
                "MRCoreset-16"
            ))
        )

    times <- select(coresets, rank, algorithm, final_tau, dataset, outliers_spec, solution=solution_time, coreset=coreset_time) %>%
        pivot_longer(solution:coreset, names_to="component", values_to="time") %>%
        mutate(time = set_units(time, "s") %>% drop_units())

    if (coreset_only) {
        times <- filter(times, component == "coreset")
    }

    sizes <- coresets %>%
        group_by(dataset, rank, algorithm, final_tau, outliers_spec) %>%
        summarise(
            coreset_size = mean(coreset_size),
            total_time = mean(solution_time + coreset_time) %>% set_units("s") %>% drop_units()
        )
    
    ggplot(times, aes(x=algorithm, y=time, fill=component)) +
        geom_bar(stat="summary") +
        geom_text(
            aes(label=coreset_size, x=algorithm),
            y=0,
            size=2,
            data=sizes,
            inherit.aes=F,
            hjust=0
        ) +
        facet_grid(vars(final_tau), vars(dataset, outliers_spec), scales="free") +
        labs(title = title) +
        coord_flip() +
        theme_paper()
}


