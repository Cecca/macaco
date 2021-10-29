plan <- drake_plan(
    higgs_data = {
        file_in("macaco-results.sqlite")
        table_dataset(c("Higgs", "Higgs-sample-10000"))
    },

    data_result = {
        file_in("macaco-results.sqlite")
        table_result()
    },

    data_memory = data_result %>%
        filter(!str_detect(dataset, "sample"), outliers_spec == 150) %>%
        drop_na(memory_coreset_bytes) %>%
        mutate(memory_coreset_mb = memory_coreset_bytes / (1024 * 1024)),

    plot_sequential_effect = do_plot_sequential_effect(data_result),
    fig_sequential_effect = ggsave("imgs/seq-effect.png", 
        plot=plot_sequential_effect,
        width=10,
        height=5
    ),
    plot_sequential_time = do_plot_sequential_time(data_result),
    fig_sequential_time = ggsave("imgs/seq-time.png", 
        plot=plot_sequential_time,
        width=10,
        height=5
    ),
    plot_mapreduce_time = do_plot_mapreduce_time(data_result),
    fig_mapreduce_time = ggsave("imgs/mr-time.png", 
        plot=plot_mapreduce_time,
        width=9,
        height=6
    ),
    plot_tradeoff = do_plot_tradeoff(data_result),
    fig_tradeoff = ggsave("imgs/tradeoff.png", 
        plot=plot_tradeoff,
        width=8,
        height=7
    ),
    plot_solution_time = do_plot_solution_time(data_result),
    fig_solution_time = ggsave("imgs/solution-time.png",
        plot = plot_solution_time,
        width = 5,
        height = 6
    ),

    plot_memory = do_plot_memory(data_memory),
    fig_memory = ggsave("imgs/memory.png",
        plot = plot_memory,
        width = 5,
        height = 5
    ),

    data_time_ratio = data_result %>%
        mutate(across(contains("time"), ~ drop_units(set_units(., "s")))) %>%
        filter(!str_detect(dataset, "sample"), str_detect(algorithm, "Coreset")) %>% 
        filter(outliers_spec == 50) %>%
        group_by(dataset, algorithm, workers, tau) %>% 
        summarise(across(c(coreset_time, solution_time), mean)) %>% 
        mutate(time_ratio = coreset_time / solution_time),
    plot_time_ratio = do_plot_time_ratio(data_time_ratio),
    fig_time_ratio = ggsave("imgs/time-ratio.png",
        plot = plot_time_ratio,
        width = 5,
        height = 4
    ),
)
