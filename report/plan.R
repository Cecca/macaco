num_outliers <- c(50, 100, 150)

plan <- drake_plan(
    higgs_data = {
        file_in("macaco-results.sqlite")
        table_dataset(c("Higgs", "Higgs-sample-10000"))
    },

    data_result = target({
            file_in("macaco-results.sqlite")
            table_result() %>%
                filter(outliers_spec == num_outliers)
        },
        transform = map(num_outliers = !!num_outliers)
    ),

    data_memory = target(
        data_result %>%
            filter(!str_detect(dataset, "sample")) %>%
            drop_na(memory_coreset_bytes) %>%
            mutate(memory_coreset_mb = memory_coreset_bytes / (1024 * 1024)),
        transform = map(data_result)
    ),

    plot_sequential_effect = target(do_plot_sequential_effect(data_result), transform=map(data_result)),
    fig_sequential_effect = target(ggsave(
            str_c("imgs/seq-effect-", outliers, ".png"), 
            plot=plot_sequential_effect,
            width=8,
            height=3
        ), 
        transform = map(plot_sequential_effect, outliers = !!num_outliers)
    ),

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
            width=9,
            height=6
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
            mutate(time_ratio = coreset_time / solution_time),
        transform = map(data_result)
    ),

    plot_time_ratio = target(do_plot_time_ratio(data_time_ratio), transform = map(data_time_ratio)),

    fig_time_ratio = target(ggsave(
            str_c("imgs/time-ratio-", outliers, ".png"),
            plot = plot_time_ratio,
            width = 5,
            height = 4
        ),
        transform = map(plot_time_ratio, outliers = !!num_outliers)
    )
)
