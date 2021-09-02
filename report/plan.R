plan <- drake_plan(
    higgs_data = {
        file_in("kcmkc-results.sqlite")
        table_dataset(c("Higgs", "Higgs-sample-10000"))
    },

    data_result = {
        file_in("kcmkc-results.sqlite")
        table_result()
    },
    plot_sequential_effect = do_plot_sequential_effect(data_result),
    fig_sequential_effect = ggsave("imgs/seq-effect.png", 
        plot=plot_sequential_effect,
        width=8,
        height=8
    ),
    plot_sequential_time = do_plot_sequential_time(data_result),
    fig_sequential_time = ggsave("imgs/seq-time.png", 
        plot=plot_sequential_time,
        width=8,
        height=8
    ),
    plot_mapreduce_time = do_plot_mapreduce_time(data_result),
    fig_mapreduce_time = ggsave("imgs/mr-time.png", 
        plot=plot_mapreduce_time,
        width=8,
        height=4
    )
    # plot_tradeoff = target(
    #     do_plot_tradeoff(data_result),
    #     # do_plot_tradeoff(filter(data_result, rank == rank_value)),
    #     # transform = cross(
    #     #     rank_value = c(10, 50)
    #     # )
    # ),
    # plot_time = target(
    #     do_plot_time(data_result, coreset_only = F),
    #     # do_plot_time(filter(data_result, rank == rank_value), coreset_only = F),
    #     # transform = cross(
    #     #     rank_value = c(10, 50)
    #     # )
    # ),
    # plot_time_coreset = target(
    #     do_plot_time(data_result, coreset_only = T),
    #     # do_plot_time(filter(data_result, rank == rank_value), coreset_only = T),
    #     # transform = cross(
    #     #     rank_value = c(10, 50)
    #     # )
    # ),
    # figure_samples = ggsave(
    #     "imgs/radius-vs-time.png",
    #     do_plot_samples(data_result),
    #     width = 10,
    #     height = 5,
    #     dpi = 300
    # )
    # notes = rmarkdown::render(
    #     knitr_in("R/notes.Rmd"),
    #     output_file = file_out("notes.html"),
    #     output_dir = "R",
    #     quiet = TRUE
    # )
)
