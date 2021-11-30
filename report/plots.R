theme_paper <- function() {
    theme_minimal() +
        theme(
            legend.position = "bottom",
            plot.margin = unit(c(0,0,0,0), "mm"),
            axis.line = element_line()
        )
}

algopalette <- c(
    # "ChenEtAl" = "#4e79a7",
    # "MRCoreset" = "#59a14f",
    # "SeqCoreset" = "#f28e2b",
    # "StreamingCoreset" = "#edc948",
    # "Random" = "#76b7b2"
    "ChenEtAl" = "#5778a4",
    "MRCoreset" = "#e49444",
    "SeqCoreset" = "#d1615d",
    "StreamingCoreset" = "#85b6b2",
    "Random" = "#e7ca60",
    "KaleStreaming" = "#a87c9f"
)

scale_color_algorithm <- function() {
    scale_color_manual(values = algopalette, aesthetics = c("fill", "color"))
}

do_plot_sequential_effect <- function(data) {
    plotdata <- data %>% 
        filter(algorithm %in% c("SeqCoreset", "StreamingCoreset", "ChenEtAl")) %>%
        group_by(dataset, rank, outliers_spec, algorithm, tau, algorithm_params) %>% 
        summarise(ratio_to_best = mean(ratio_to_best)) %>%
        filter(!str_detect(dataset, "sample")) %>%
        mutate(
            sample = if_else(str_detect(dataset, "sample"), "sample", "full"),
            dataset = str_remove(dataset, "-sample-10000")
        )
    # plotdata %>% ungroup() %>% distinct(dataset) %>% print()

    doplot <- function(plotdata, titlestr) {
        # baseline <- plotdata %>% filter(algorithm == "ChenEtAl")
        # kale <- plotdata %>% 
        #     filter(algorithm == 'KaleStreaming') %>%
        #     group_by(dataset, rank, outliers_spec) %>%
        #     slice_min(ratio_to_best)
        # print(kale)
        plotdata <- plotdata %>% filter(tau <= 10)
        p <- ggplot(plotdata, aes(x=tau, y=ratio_to_best, color=algorithm)) +
            geom_point() +
            geom_line() +
            # geom_hline(
            #     aes(yintercept=ratio_to_best),
            #     data=baseline,
            #     color=algopalette['ChenEtAl']
            # ) +
            # geom_hline(
            #     aes(yintercept = ratio_to_best),
            #     data = kale,
            #     color = algopalette['KaleStreaming'],
            #     linetype = "solid",
            #     size = 1
            # ) +
            facet_wrap(vars(dataset), scales="free_y") +
            scale_x_continuous(breaks=scales::pretty_breaks()) +
            scale_color_algorithm() +
            # coord_cartesian(ylim=c(0, NA)) +
            labs(
                title=titlestr,
                x = TeX("$\\tau$")
            ) +
            theme_paper() +
            theme(
                legend.position = "none"
            ) 

        p
    }

    doplot(filter(plotdata, sample == "full"), "")

}

do_plot_streaming <- function(data) {
    plotdata <- data %>%
        filter(algorithm %in% c("StreamingCoreset", "KaleStreaming")) %>%
        filter(!str_detect(dataset, "sample")) %>%
        rowwise() %>%
        mutate(epsilon = access_json(algorithm_params, "epsilon")) %>%
        ungroup() %>%
        group_by(dataset, rank, outliers_spec, algorithm, epsilon, tau, algorithm_params) %>%
        summarise(
            ratio_to_best = mean(ratio_to_best),
            total_time = set_units(mean(total_time), "s") %>% drop_units(),
            memory_coreset_mb = mean(memory_coreset_mb)
        ) %>%
        arrange(tau, epsilon) %>%
        ungroup()

    draw <- function(plotdata, y, xlab, ylab) {
        p <- ggplot(plotdata, aes(x = memory_coreset_mb, y = {{ y }}, color=algorithm)) +
            geom_point() +
            # geom_path() +
            scale_color_algorithm() +
            facet_wrap(vars(dataset), scales="free_y") +
            labs(
                x = xlab,
                y = ylab
            ) +
            theme_paper() +
            theme(legend.position="none")

        if (length(xlab) == 0) {
            p <- p + theme(axis.title.x = element_none())
        }

        p
    }

    draw(plotdata, ratio_to_best, "", "Ratio to best") / 
        draw(plotdata, total_time, "Memory (MB)", "Total time") 
        
}

do_plot_all_sequential <- function(data) {
    plotdata <- data %>%
        filter(!(algorithm %in% c("MRCoreset", "KaleStreaming"))) %>%
        filter(!(str_detect(dataset, "sample"))) %>%
        group_by(dataset, rank, outliers_spec, 
                 algorithm, tau, algorithm_params) %>%
        summarise(
            ratio_to_best = mean(ratio_to_best),
            total_time = set_units(mean(total_time), "s") %>% drop_units()
        ) %>%
        ungroup()

    draw <- function(plotdata, y, xlab, ylab, trans_y = "identity") {
        plotdata <- filter(plotdata, tau <= 10)
        p <- ggplot(plotdata, aes(x = tau, y = {{ y }}, color=algorithm)) +
            geom_point() +
            geom_path() +
            scale_color_algorithm() +
            scale_y_continuous(trans = trans_y) +
            scale_x_continuous(breaks = scales::pretty_breaks()) +
            facet_wrap(vars(dataset), scales="free_y") +
            labs(
                x = xlab,
                y = ylab
            ) +
            theme_paper() +
            theme(legend.position="none")

        if (length(xlab) == 0) {
            p <- p + theme(axis.title.x = element_none())
        }

        p
    }

    draw(plotdata, ratio_to_best, "", "Ratio to best") /
        draw(plotdata, total_time, TeX("$\\tau$"), "Total time", trans_y = "identity")

}



do_plot_sequential_effect_artificial_outliers <- function(data) {
    plotdata <- data %>% 
        filter(algorithm %in% c("SeqCoreset", "StreamingCoreset", "ChenEtAl", "KaleStreaming")) %>%
        filter(algorithm == "SeqCoreset") %>%
        # filter( tau > 70) %>%
        group_by(dataset, rank, outliers_spec, algorithm, tau, algorithm_params) %>% 
        # summarise(ratio_to_best = mean(ratio_to_best)) %>%
        mutate(
            sample = if_else(str_detect(dataset, "sample"), "sample", "full"),
            dataset = str_remove(dataset, "-sample-10000")
        )

    doplot <- function(plotdata, titlestr) {
        baseline <- plotdata %>% filter(algorithm == "ChenEtAl")
        kale <- plotdata %>% 
            filter(algorithm == 'KaleStreaming') %>%
            group_by(dataset, rank, outliers_spec) %>%
            slice_min(ratio_to_best)
        p <- ggplot(plotdata, aes(x=tau, y=ratio_to_best, color=algorithm)) +
            # geom_text(aes(x = 1, y=50), label="arbitrary independent set", nudge_x=2, hjust=0, inherit.aes=F, linetype="dotted") +
            geom_vline(aes(xintercept = outliers_spec + 1), inherit.aes=F, linetype="dotted") +
            geom_text(aes(x = outliers_spec + 1, y=50), label="z + 1", nudge_x=1, hjust=0, inherit.aes=F, linetype="dotted") +
            geom_vline(aes(xintercept = rank + outliers_spec), inherit.aes=F, linetype="dashed") +
            geom_text(aes(x =rank + outliers_spec + 1, y=50), nudge_x=1, label="k + z", hjust=0, inherit.aes=F, linetype="dotted") +
            stat_summary(geom="point", size=0.2) +
            geom_point(size=0.2, alpha=0.5) +
            geom_line(stat="summary") +
            geom_hline(
                aes(yintercept=ratio_to_best),
                data=baseline,
                color=algopalette['ChenEtAl']
            ) +
            geom_hline(
                aes(yintercept = ratio_to_best),
                data = kale,
                color = algopalette['KaleStreaming'],
                linetype = "solid",
                size = 1
            ) +
            # facet_wrap(vars(dataset), scales="free_y") +
            scale_x_continuous(breaks=scales::pretty_breaks()) +
            scale_color_algorithm() +
            # coord_cartesian(ylim=c(0, NA)) +
            labs(
                title=titlestr,
                x = TeX("$\\tau$"),
                y = "Ratio to best"
            ) +
            theme_paper() +
            theme(
                panel.border = element_rect(fill=NA),
                legend.position = "none"
            )

        p
    }

    doplot(filter(plotdata, sample == "full"), "")

}

do_plot_sequential_coreset_size <- function(data) {
    plotdata <- data %>% 
        filter(algorithm %in% c("SeqCoreset", "StreamingCoreset", "ChenEtAl", "KaleStreaming")) %>%
        group_by(dataset, rank, outliers_spec, algorithm, tau, algorithm_params) %>% 
        summarise(
            ratio_to_best = mean(ratio_to_best),
            coreset_size = mean(coreset_size)
        ) %>%
        mutate(
            sample = if_else(str_detect(dataset, "sample"), "sample", "full"),
            dataset = str_remove(dataset, "-sample-10000")
        )


    doplot <- function(plotdata, titlestr) {
        plotdata %>% ungroup() %>% filter(dataset == "Higgs-z50") %>% print()
        baseline <- plotdata %>% filter(algorithm == "ChenEtAl")
        kale <- plotdata %>% 
            filter(algorithm == 'KaleStreaming') %>%
            group_by(dataset, rank, outliers_spec) %>%
            slice_min(ratio_to_best)
        #plotdata <- plotdata %>% filter(tau <= 10)
        p <- ggplot(plotdata, aes(x=tau, y=coreset_size, color=algorithm)) +
            geom_point() +
            geom_line() +
            geom_hline(
                aes(yintercept=coreset_size),
                data=baseline,
                color=algopalette['ChenEtAl']
            ) +
            geom_hline(
                aes(yintercept = coreset_size),
                data = kale,
                color = algopalette['KaleStreaming'],
                linetype = "solid",
                size = 1
            ) +
            facet_wrap(vars(dataset), scales="free_y") +
            scale_x_continuous(breaks=scales::pretty_breaks()) +
            scale_y_continuous(trans = "log10") +
            scale_color_algorithm() +
            # coord_cartesian(ylim=c(0, NA)) +
            labs(title=titlestr) +
            theme_paper() +
            theme(
                panel.border = element_rect(fill=NA)
            ) +
            labs(
            )

        p
    }

    doplot(filter(plotdata, sample == "full"), "Full data") |
        doplot(filter(plotdata, sample == "sample"), "Sampled data")

}

do_coreset_construction_comparision <- function(data) {
    plotdata <- data %>% 
        filter(!str_detect(dataset, "sample")) %>% 
        filter(algorithm %in% c("KaleStreaming", "SeqCoreset", "StreamingCoreset")) %>% 
        mutate(coreset_time = drop_units(set_units(coreset_time, "s"))) %>%
        group_by(dataset, outliers_spec, rank, algorithm, algorithm_params) %>%
        summarise(across(matches("coreset*"), mean))
    print(plotdata)

    ggplot(plotdata, aes(coreset_size, 
                         coreset_time, 
                         color=algorithm, 
                         shape=dataset)) +
        geom_point() + 
        scale_color_algorithm() +
        scale_x_continuous(trans="log10") + 
        scale_y_continuous(trans="log10") + 
        # facet_grid(vars(dataset), vars(outliers_spec)) +
        theme_paper() + 
        theme(legend.position="right") 
}

do_plot_sequential_time <- function(data) {
    plotdata <- data %>% 
        filter(algorithm %in% c("SeqCoreset", "StreamingCoreset", "ChenEtAl", "KaleStreaming")) %>%
        group_by(dataset, rank, outliers_spec, algorithm, tau, algorithm_params) %>% 
        summarise(
            ratio_to_best = mean(ratio_to_best),
            total_time = if_else(algorithm == "KaleStreaming",
                mean(coreset_time + solution_time),
                mean(total_time),
            ) %>% set_units("s") %>% drop_units()
        ) %>%
        mutate(
            sample = if_else(str_detect(dataset, "sample"), "sample", "full"),
            dataset = str_remove(dataset, "-sample-10000")
        )

    doplot <- function(plotdata, titlestr) {
        baseline <- plotdata %>% filter(algorithm == "ChenEtAl")
        kale <- plotdata %>% 
            filter(algorithm == 'KaleStreaming') %>%
            group_by(dataset, rank, outliers_spec) %>%
            slice_min(ratio_to_best)
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
            geom_hline(
                aes(yintercept = total_time),
                data = kale,
                color = algopalette['KaleStreaming'],
                linetype = "solid"
            ) +
            facet_wrap(vars(dataset), scales="free_y") +
            scale_y_log10(labels=scales::number_format(accuracy=1)) +
            scale_x_continuous(breaks=scales::pretty_breaks()) +
            scale_color_algorithm() +
            labs(title = titlestr) +
            theme_paper() +
            theme(
                panel.border = element_rect(fill=NA)
            ) +
            labs(
                y = "total time (s)"
            )

        p
    }

    doplot(filter(plotdata, sample == "full"), "Full dataset") |
        doplot(filter(plotdata, sample == "sample"), "Sampled dataset")
}

do_plot_mapreduce_time <- function(data) {
    data <- data %>%
        filter(tau %in% c(3, 6, 9)) %>%
        filter(!str_detect(dataset, "sample")) %>%
        mutate(across(contains("_time"), ~ set_units(.x, "s") %>% drop_units())) %>%
        mutate(tau = str_c("τ=", tau))

    mr <- data %>% filter(algorithm == "MRCoreset") %>%
        group_by(dataset, rank, outliers_spec, workers, tau) %>%
        summarise(across(contains("_time"), mean))
    seq <- data %>% filter(algorithm == "SeqCoreset") %>%
        group_by(dataset, rank, outliers_spec, tau) %>%
        summarise(across(contains("_time"), mean))

    p <- ggplot(mr, aes(x=workers, y=coreset_time)) +
        geom_segment(aes(xend=workers, y=coreset_time, yend=0)) +
        geom_point(aes(y=coreset_time), show.legend=F) +
        geom_hline(aes(yintercept=coreset_time), data=seq, linetype="dotted") +
        # geom_segment(aes(xend=tau, y=coreset_time, yend=coreset_time + solution_time), color="red") +
        facet_grid(vars(dataset), vars(tau), scales="free") +
        scale_color_algorithm() +
        scale_x_continuous(
            trans="log2", 
            limits=c(0.5, 16),
            # expand=expansion(mult=c(1,1)),
            breaks=c(1,2,4,8)
        ) +
        scale_y_continuous(trans="identity") +
        theme_paper() +
        theme(
            panel.grid = element_blank(),
            panel.border = element_blank(),
            axis.line.x = element_blank()
        ) +
        coord_cartesian(clip="on",) +
        labs(
            y = "coreset construction time (s)"
        ) +
        annotate(geom="segment", x=0.5, xend=16, y=0, yend=0)

    p
}

do_plot_solution_time <- function(data) {
    plotdata <- data %>% 
        group_by(dataset, outliers_spec, workers, algorithm, tau) %>% 
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
        group_by(dataset, outliers_spec) %>% 
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
            outliers_spec = outliers_spec,
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
    print(reflines)

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
        filter(
            (str_detect(algorithm, "MRCoreset") & (workers == 8)) | (algorithm != "MRCoreset"),
            tau %in% c(3,6,9)
        ) %>%
        mutate(dominant = if_else(time_ratio < 1, "#ef8a62", "#67a9cf"))
        # filter(algorithm != "MRCoreset")

    # plotdata %>% filter(algorithm == "MRCoreset", dataset=="Higgs") %>% print()

    plot_one <- function(plotdata, title, strip=F, ylab=NULL) {
        if (nrow(plotdata) == 0) {
            return(ggplot())
        }
        p <- ggplot(plotdata, aes(x = tau, color=dominant)) +
            geom_point(
                aes(y = coreset_time),
                color = "#67a9cf"
            ) +
            geom_point(
                aes(y = -solution_time),
                color = "#ef8a62"
            ) +
            geom_linerange(
                aes(ymax = coreset_time),
                ymin = 0,
                color = "#67a9cf"
            ) +
            geom_linerange(
                aes(ymin = -solution_time),
                ymax = 0,
                color = "#ef8a62"
            ) +
            # geom_segment(aes(yend=1, xend=tau)) +
            geom_hline(yintercept=0) +
            # geom_vline(xintercept =0) +
            scale_y_continuous(
                trans = "identity",
                labels = abs
                # breaks=c(0.01, 0.1, 1, 10, 100),
                # labels=c("", "10", "1", "10", "")
            ) +
            scale_x_continuous(
                breaks = c(3,6,9)
            ) +
            scale_color_identity() +
            facet_wrap(
                vars(dataset), 
                scales = "free",
                ncol = 1,
                strip.position = "right"
            ) +
            labs(
                title = title,
                x = ylab,
                y = "Time (s)"
            ) +
            coord_flip(clip="off") +
            theme_paper() +
            theme(
                panel.grid = element_blank(),
                panel.grid.major.x = element_line(size=0.2, color="lightgray"),
                axis.line.y = element_blank(),
                axis.line.x = element_blank(),
                plot.title = element_text(size = 8),
                strip.text = element_text(size = 8),
                strip.background = element_rect(),
                text = element_text(size = 8),
                panel.spacing = unit(4, "mm")
            )
        # if (!strip) {
        p <- p + theme(
            strip.text = element_blank()
        )
        # }
        if (strip) {
            # Add the strip annotation. We use this trick because facet labels are clipped
            labels <- plotdata %>%
                group_by(dataset) %>%
                summarise(t = max(coreset_time) + max(coreset_time + solution_time) * 0.2)
            p <- p +
                geom_text(
                    aes(label = dataset, y = t),
                    data = labels,
                    inherit.aes = F,
                    x = 6,
                    angle = 270,
                    size = 3
                )
                #scale_y_continuous(expand = )
        }
        if (is.null(ylab)) {
            p <- p + theme(
                axis.text.y = element_blank()
            )
        }

        p
    }

    plot_one(filter(plotdata, algorithm == "SeqCoreset"), "SeqCoreset", ylab=TeX("$\\tau$")) | 
        (plot_one(filter(plotdata, algorithm == "StreamingCoreset"), "StreamingCoreset") + theme(plot.margin = unit(c(0, 4, 0, 8), "mm"))) |
        (plot_one(filter(plotdata, algorithm == "MRCoreset"), "MRCoreset", strip=F) + theme(plot.margin = unit(c(0, 8, 0, 4), "mm"))) |
        plot_one(filter(plotdata, algorithm == "MRCoresetRec"), "MRCoresetRec", strip=T)
}

do_plot_tradeoff <- function(data) {
    plotdata <- data %>%
        filter(algorithm %in% c("StreamingCoreset", "SeqCoreset")) %>%
        mutate(total_time = set_units(total_time, "s") %>% drop_units()) %>%
        mutate(dataset = fct_reorder(dataset, is_sample))

    averages <- plotdata %>%
        mutate(algorithm_params = if_else(algorithm == "Random", "", algorithm_params)) %>%
        group_by(dataset, rank, workers, outliers_spec, algorithm, algorithm_params) %>%
        summarise(radius = mean(radius), ratio_to_best = mean(ratio_to_best), total_time = mean(total_time), coreset_size = mean(coreset_size)) %>%
        ungroup()

    baselines <- data %>%
        filter(algorithm %in% c("KaleStreaming", "ChenEtAl")) %>%
        mutate(total_time = set_units(total_time, "s") %>% drop_units()) %>%
        mutate(dataset = fct_reorder(dataset, is_sample)) %>%
        group_by(dataset, rank, workers, outliers_spec, algorithm, algorithm_params) %>%
        summarise(radius = mean(radius), ratio_to_best = mean(ratio_to_best), total_time = mean(total_time), coreset_size = mean(coreset_size)) %>%
        # Get the configuration running fastest
        slice_min(total_time) %>%
        ungroup()

    draw <- function(data) {
        data %>% distinct(dataset, rank) %>% print()
        title <- data %>% distinct(dataset) %>% pull()
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
                aes(xintercept = ratio_to_best),
                data = baselines
            ) +
            geom_hline(
                aes(yintercept = total_time),
                data = baselines
            ) +
            geom_point(size = 1.2, stat="summary", fun.data=mean_se) +
            scale_y_continuous(trans = "log10") +
            scale_color_algorithm() +
            labs(y = "total time (s)", x = "radius", title=title) +
            # facet_grid(vars(dataset), vars(rank), scales = "free") +
            facet_wrap(vars(dataset), scales = "free", ncol=1) +
            theme_paper()
    }

    averages %>% 
        filter(dataset %in% c("Higgs", "Phones", "Wikipedia")) %>% 
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

do_plot_memory <- function(plotdata) {
    plotdata <- plotdata %>% 
        rowwise() %>%
        mutate(epsilon = access_json(algorithm_params, "epsilon")) %>%
        ungroup() %>%
        filter(is.na(epsilon) | (epsilon %in% c(0.5, 1, 2)))
    sequential <- plotdata %>% 
        filter(algorithm == "SeqCoreset") %>%
        group_by(dataset, outliers_spec) %>%
        slice_min(memory_coreset_mb)
    print(sequential)
    plotdata <- plotdata %>% filter(algorithm != "SeqCoreset")
    ggplot(plotdata, aes(x = tau, 
                         y=memory_coreset_mb, 
                         color=algorithm)) +
        geom_point(
            data = ~ filter(.x, algorithm == "StreamingCoreset"),
            stat="summary"
        ) +
        geom_line(
            data = ~ filter(.x, algorithm == "StreamingCoreset"),
            stat="summary"
        ) +
        geom_hline(
            data = ~ filter(.x, algorithm == "KaleStreaming"),
            mapping = aes(
                yintercept=memory_coreset_mb
            ),
            linetype = "dotted",
            color = algopalette["KaleStreaming"]
        ) +
        geom_text(
            data = ~ filter(.x, algorithm == "KaleStreaming"),
            mapping = aes(label = str_c("ε = ", epsilon)),
            color = algopalette["KaleStreaming"],
            nudge_y = 0.1,
            hjust = 0,
            vjust = 0,
            size = 4,
            x = 1
        ) +
        geom_text(
            data = sequential,
            mapping = aes(label = scales::number(
                                        memory_coreset_mb, 
                                        accuracy = 1,
                                        prefix="SeqCoreset:\n", 
                                        suffix="MB")),
            color = algopalette["SeqCoreset"],
            hjust = 1,
            vjust = 0,
            size = 4,
            x = 9,
            y = 5
        ) +
        # facet_grid(vars(algorithm), vars(dataset), scales="free_y") +
        facet_wrap(vars(dataset)) +
        scale_color_algorithm() +
        # scale_y_log10() +
        coord_cartesian(ylim=c(0, 6)) +
        labs(
            x = TeX("$\\tau$"),
            y = "Memory (MB)"
        ) +
        theme_paper()
}

do_plot_final_approximation <- function(plotdata) {
    plotdata <- plotdata %>% 
        filter(!str_detect(dataset, "sample")) %>% 
        filter(is.na(tau) | (tau > 1)) %>%
        filter(algorithm %in% c("KaleStreaming", "SeqCoreset", "StreamingCoreset", "MRCoreset")) %>% 
        filter(outliers_spec == 50) %>%
        group_by(dataset, outliers_spec, rank, algorithm, algorithm_params) %>%
        summarise(
            ratio_to_best = mean(ratio_to_best),
            coreset_size = mean(coreset_size))
    print(plotdata)

    ggplot(plotdata, aes(coreset_size, 
                         ratio_to_best, 
                         color=algorithm)) +
        geom_point() + 
        scale_color_algorithm() +
        scale_x_continuous(trans="log10") + 
        scale_y_continuous(trans="identity") + 
        facet_wrap(vars(dataset)) +
        theme_paper() + 
        theme(legend.position="right") 

}
