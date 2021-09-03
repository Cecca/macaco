theme_paper <- function() {
    theme_minimal() +
        theme(
            legend.position = "bottom",
            axis.line = element_line()
        )
}

algopalette <- c(
    "ChenEtAl" = "#4e79a7",
    "MRCoreset" = "#59a14f",
    "SeqCoreset" = "#f28e2b",
    "StreamingCoreset" = "#edc948",
    "Random" = "#76b7b2"
)

scale_color_algorithm <- function() {
    scale_color_manual(values = algopalette, aesthetics = c("fill", "color"))
}

do_plot_sequential_effect <- function(data) {
    plotdata <- data %>% 
        filter(algorithm %in% c("SeqCoreset", "StreamingCoreset", "ChenEtAl")) %>%
        group_by(dataset, rank, outliers_spec, algorithm, tau) %>% 
        summarise(ratio_to_best = mean(ratio_to_best)) %>%
        mutate(
            sample = if_else(str_detect(dataset, "sample"), "sample", "full"),
            dataset = str_remove(dataset, "-sample-10000")
        )
    baseline <- plotdata %>% filter(algorithm == "ChenEtAl")
    plotdata <- plotdata %>% filter(tau <= 10)

    p <- ggplot(plotdata, aes(x=tau, y=ratio_to_best, color=algorithm)) +
        geom_point() +
        geom_hline(
            aes(yintercept=ratio_to_best),
            data=baseline,
            color=algopalette['ChenEtAl']
        ) +
        facet_grid(vars(dataset), vars(sample), scales="free_y") +
        scale_x_continuous(breaks=scales::pretty_breaks()) +
        scale_color_algorithm() +
        theme_paper() +
        theme(
            panel.border = element_rect(fill=NA)
        ) +
        labs(
        )

    p
}

do_plot_sequential_time <- function(data) {
    plotdata <- data %>% 
        filter(algorithm %in% c("SeqCoreset", "StreamingCoreset", "ChenEtAl")) %>%
        group_by(dataset, rank, outliers_spec, algorithm, tau) %>% 
        summarise(total_time = mean(total_time) %>% set_units("s") %>% drop_units()) %>%
        mutate(
            sample = if_else(str_detect(dataset, "sample"), "sample", "full"),
            dataset = str_remove(dataset, "-sample-10000")
        )
    baseline <- plotdata %>% filter(algorithm == "ChenEtAl")
    plotdata <- plotdata %>% filter(tau <= 10)

    p <- ggplot(
            filter(plotdata, algorithm != "ChenEtAl"), 
            aes(x=tau, y=total_time, color=algorithm)
        ) +
        geom_line() +
        geom_point() +
        geom_hline(
            aes(yintercept=total_time),
            data=baseline,
            color=algopalette['ChenEtAl']
        ) +
        facet_grid(vars(dataset), vars(sample), scales="free_y") +
        scale_y_log10(labels=scales::number_format(accuracy=1)) +
        scale_x_continuous(breaks=scales::pretty_breaks()) +
        scale_color_algorithm() +
        theme_paper() +
        theme(
            panel.border = element_rect(fill=NA)
        ) +
        labs(
            y = "total time (s)"
        )

    p
}

do_plot_mapreduce_time <- function(data) {
    data <- data %>%
        filter(tau <= 10) %>%
        filter(!str_detect(dataset, "sample")) %>%
        mutate(across(contains("_time"), ~ set_units(.x, "s") %>% drop_units()))
    mr <- data %>% filter(algorithm == "MRCoreset") %>%
        group_by(dataset, rank, outliers_spec, workers, tau) %>%
        summarise(across(contains("_time"), mean))
    seq <- data %>% filter(algorithm == "SeqCoreset") %>%
        group_by(dataset, rank, outliers_spec, tau) %>%
        summarise(across(contains("_time"), mean))

    p <- ggplot(mr, aes(x=workers, y=coreset_time)) +
        geom_segment(aes(xend=workers, y=coreset_time, yend=0)) +
        geom_point(aes(y=coreset_time, shape=factor(workers)), show.legend=F) +
        geom_hline(aes(yintercept=coreset_time), data=seq, linetype="dotted") +
        # geom_segment(aes(xend=tau, y=coreset_time, yend=coreset_time + solution_time), color="red") +
        facet_grid(vars(dataset), vars(tau), scales="free") +
        scale_color_algorithm() +
        scale_x_continuous(
            trans="log2", 
            limits=c(1, 16),
            # expand=expansion(mult=c(1,1)),
            breaks=c(2,4,8)
        ) +
        scale_y_continuous(trans="log2") +
        theme_paper() +
        theme(
            panel.grid = element_blank(),
            panel.border = element_blank(),
            axis.line.x = element_blank()
        ) +
        coord_cartesian(clip="on",) +
        labs(
            y = "total time (s)"
        ) +
        annotate(geom="segment", x=1, xend=16, y=0, yend=0)

    p
}

do_plot_solution_time <- function(data) {
    plotdata <- data %>% 
        group_by(dataset, workers, algorithm, tau) %>% 
        summarise(across(c(solution_time, coreset_size), ~ mean(.x))) %>% 
        filter(!str_detect(dataset, "sample")) %>%
        filter(tau %in% c(3, 6, 10)) %>%
        mutate(solution_time = drop_units(set_units(solution_time, "s")))

    # mincsize <- ungroup(plotdata) %>% summarise(min(coreset_size)) %>% pull()
    # mintime <- ungroup(plotdata) %>% summarise(min(solution_time)) %>% pull()
    # scalelin <- mintime / mincsize
    # scalequad <- mintime / mincsize^2
    # scalecube <- mintime / mincsize^3

    reflines <- plotdata %>% 
        group_by(dataset) %>% 
        summarise(
            mintime = min(solution_time), 
            minsize = min(coreset_size), 
            scale_lin = mintime / minsize, 
            scale_quad = mintime / minsize^2, 
            scale_cub = mintime / minsize^3
        )  %>%
        rowwise() %>% 
        summarise(
            dataset = dataset,
            tribble(
                ~label, ~x, ~y, 
                "n", minsize, scale_lin*minsize, 
                "n", 1000, 1000*scale_lin,
                "n^2", minsize, scale_quad*minsize^2,
                "n^2", 1000, 1000^2*scale_quad,
                "n^3", minsize, scale_cub*minsize^3,
                "n^3", 1000, 1000^3*scale_cub,
            )
        )

    ggplot(plotdata, aes(coreset_size, solution_time, shape=factor(tau), color=algorithm)) +
        geom_line(
            aes(x, y, group=label),
            data=reflines,
            inherit.aes=F,
            linetype="dotted"
        ) +
        geom_text(
            aes(x, y, label=label),
            data=filter(reflines, x == 1000),
            inherit.aes=F,
            vjust=0,
            hjust=1,
            size=3,
            parse=T
        ) +
        geom_point() +
        scale_y_continuous(
            labels=scales::number_format(), 
            breaks=scales::breaks_log(),
            trans="log"
        ) + 
        scale_x_continuous(
            labels=scales::number_format(), 
            breaks=c(100, 300, 1000),
            trans="log"
        ) + 
        labs(
            x="coreset size",
            y="time (s)",
            shape=TeX("\\tau")
        ) +
        facet_wrap(vars(dataset)) +
        coord_cartesian(clip="off") +
        theme_paper() +
        theme(
            panel.grid = element_blank()
        )
}

do_plot_time_ratio <- function(data) {
    plotdata <- data %>%
        filter(workers %in% c(1, 8))

    ggplot(plotdata, aes(tau, time_ratio)) +
        geom_point() +
        geom_segment(aes(yend=1, xend=tau)) +
        geom_hline(yintercept=1) +
        scale_y_log10(labels=scales::number_format(accuracy=0.01)) +
        facet_grid(vars(dataset), vars(algorithm)) +
        theme_paper()
}

do_plot_tradeoff <- function(data) {
    plotdata <- data %>%
        mutate(total_time = set_units(total_time, "s") %>% drop_units()) %>%
        mutate(dataset = fct_reorder(dataset, is_sample))

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
            geom_point(size = 1.2, stat="summary", fun.data=mean_se) +
            # scale_y_continuous(trans = "log10") +
            scale_color_algorithm() +
            labs(y = "total time (s)", x = "radius", title=title) +
            # facet_grid(vars(dataset), vars(rank), scales = "free") +
            facet_wrap(vars(dataset, rank), scales = "free", ncol=1) +
            theme_paper()
    }

    averages %>% 
        filter(dataset %in% c("Higgs", "MusixMatch", "Wikipedia")) %>% 
        draw()
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
        # filter(outliers_spec == "Percentage(0.01)") %>%
        filter(dataset %in% c("MusixMatch-sample-10000", "Wikipedia-sample-10000", "Random")) %>%
        filter(rank %in% c(20, 10)) %>%
        group_by(dataset, algorithm, rank, dataset_params, algorithm_params, outliers_spec, tau, workers) %>%
        summarise(
            total_time = mean(total_time),
            ratio_to_best = mean(ratio_to_best)
        ) %>%
        mutate(flabel = str_c(dataset, " (rank ", rank, ", outliers ", outliers_spec, ")")) %>%
        ggplot(aes(ratio_to_best, total_time, color=algorithm)) +
        geom_point() +
        geom_text_repel(aes(label = tau), show.legend=F) +
        scale_y_unit(trans="log10", labels=scales::number_format()) +
        scale_color_algorithm() +
        facet_wrap(vars(flabel), scales="free") +
        theme_paper()
}
