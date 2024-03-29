num_outliers <- c(50, 100, 150)

plan <- drake_plan(
    higgs_data = {
        file_in("macaco-results.sqlite")
        table_dataset(c("Higgs", "Higgs-sample-10000"))
    },

    data_result = target({
            file_in("macaco-results.sqlite")
            table_result() %>%
                filter(outliers_spec == num_outliers) %>%
                filter(!str_detect(dataset, "-z50"))
        },
        transform = map(num_outliers = !!num_outliers)
    ),

    data_outliers_pilot = target({
            file_in("macaco-results.sqlite")
            table_result() %>%
                filter(outliers_spec == 50) %>%
                filter(str_detect(dataset, "-z50")) %>%
                filter(tau != 2) %>%
                filter(dataset == "Higgs-z50")
        }
        #transform = map(num_outliers = !!num_outliers)
    ),

    data_memory = target(
        data_result %>%
            filter(!str_detect(dataset, "sample")) %>%
            drop_na(memory_coreset_bytes) %>%
            mutate(memory_coreset_mb = memory_coreset_bytes / (1024 * 1024)),
        transform = map(data_result)
    ),

    latex_sample = target(
        do_latex_sample(data_result),# %>% write_file(str_c("imgs/sample-", outliers, ".tex")),
        transform = map(data_result, outliers = !!num_outliers)
    ),

    plot_streaming = target(do_plot_streaming(data_memory), transform=map(data_memory)),
    fig_streaming = target(ggsave(
            str_c("imgs/streaming-", outliers, ".png"), 
            plot=plot_streaming,
            width=4,
            height=4
        ), 
        transform = map(plot_streaming, outliers = !!num_outliers)
    ),

    # Plot the performance on the samples of datasets
    plot_all_sequential = target(do_plot_all_sequential(data_result), transform=map(data_result)),
    fig_all_sequential = target(ggsave(
            str_c("imgs/sequential-", outliers, ".png"), 
            plot=plot_all_sequential,
            width=4,
            height=4
        ), 
        transform = map(plot_all_sequential, outliers = !!num_outliers)
    ),


    plot_sequential_effect = target(do_plot_sequential_effect(data_result), transform=map(data_result)),
    fig_sequential_effect = target(ggsave(
            str_c("imgs/seq-effect-", outliers, ".png"), 
            plot=plot_sequential_effect,
            width=4,
            height=3
        ), 
        transform = map(plot_sequential_effect, outliers = !!num_outliers)
    ),

    plot_seq_effect_pilot = target(do_plot_sequential_effect_artificial_outliers(data_outliers_pilot)),
    fig_seq_effect_pilot = target(ggsave(
        "imgs/seq-effect-pilot-z50.png",
        plot=plot_seq_effect_pilot,
        width=4,
        height=2
    )),

    plot_sequential_time = target(do_plot_sequential_time(data_result), transform=map(data_result)),
    fig_sequential_time = target(ggsave(
            str_c("imgs/seq-time-", outliers, ".png"),
            plot=plot_sequential_time,
            width=8,
            height=3
        ),
        transform = map(plot_sequential_time, outliers = !!num_outliers)
    ),

    plot_sequential_coreset_size = target(do_plot_sequential_coreset_size(data_result), transform = map(data_result)),
    fig_sequential_coreset_size = target(ggsave(
            str_c("imgs/seq-coreset-size-", outliers, ".png"), 
            plot=plot_sequential_coreset_size,
            width=4,
            height=6
        ), 
        transform = map(plot_sequential_coreset_size, outliers = !!num_outliers)
    ),

    # plot_coreset_construction_comparison = do_coreset_construction_comparision(data_result),
    # fig_coreset_construction_comparison = ggsave("imgs/coreset-construction.png", 
    #     plot=plot_coreset_construction_comparison,
    #     width=10,
    #     height=5
    # ),

    # plot_final_approximation = do_plot_final_approximation(data_result),
    # fig_final_approximation = ggsave("imgs/final-approximation.png",
    #     plot=plot_final_approximation,
    #     width=10,
    #     height=5
    # ),

    plot_mapreduce_time = target(do_plot_mapreduce_time(data_result), transform = map(data_result)),
    fig_mapreduce_time = target(ggsave(
            str_c("imgs/mr-time-", outliers, ".png"), 
            plot=plot_mapreduce_time,
            width=4,
            height=3
        ),
        transform = map(plot_mapreduce_time, outliers = !!num_outliers)
    ),
    plot_tradeoff = target(do_plot_tradeoff(data_result), transform = map(data_result)),
    fig_tradeoff = target(ggsave(
            str_c("imgs/tradeoff-", outliers, ".png"),
            plot=plot_tradeoff,
            width=8,
            height=7
        ),
        transform = map(plot_tradeoff, outliers = !!num_outliers)
    ),
    # plot_solution_time = do_plot_solution_time(data_result),
    # fig_solution_time = ggsave("imgs/solution-time.png",
    #     plot = plot_solution_time,
    #     width = 5,
    #     height = 6
    # ),

    plot_memory = target(do_plot_memory(data_memory), transform = map(data_memory)),
    fig_memory = target(ggsave(
            str_c("imgs/memory-", outliers, ".png"),
            plot = plot_memory,
            width = 5,
            height = 4
        ),
        transform = map(plot_memory, outliers = !!num_outliers)
    ),

    data_time_ratio = target(
        data_result %>%
            mutate(across(contains("time"), ~ drop_units(set_units(., "s")))) %>%
            filter(!str_detect(dataset, "sample"), str_detect(algorithm, "Coreset")) %>% 
            group_by(dataset, algorithm, workers, tau) %>% 
            summarise(across(c(coreset_time, solution_time), mean)) %>% 
            mutate(time_ratio = coreset_time / solution_time) %>%
            ungroup(),
        transform = map(data_result)
    ),

    plot_time_ratio = target(do_plot_time_ratio(data_time_ratio), transform = map(data_time_ratio)),

    fig_time_ratio = target(ggsave(
            str_c("imgs/time-ratio-", outliers, ".png"),
            plot = plot_time_ratio,
            width = 8,
            height = 2
        ),
        transform = map(plot_time_ratio, outliers = !!num_outliers)
    ),

    # The approximation ratios of MRCoresetRec against SeqCoreset
    mapr_rec_ratio = target(
        data_result %>%
            filter(!str_detect(dataset, "sample")) %>%
            filter(((algorithm == "MRCoresetRec" & workers==8) | (algorithm == "SeqCoreset"))) %>%
            group_by(dataset, algorithm, tau) %>%
            summarise(
                ratio_to_best = scales::number(mean(ratio_to_best), accuracy = 0.001)
            ) %>%
            pivot_wider(names_from=algorithm, values_from=ratio_to_best) %>%
            drop_na() %>%
            kbl(format="latex", booktabs=T, linesep="") %>%
            kable_styling() %>%
            write_file(str_c("imgs/mr-rec-approx-", outliers, ".tex")),
        transform = map(data_result, outliers = !!num_outliers)
    ),
)
