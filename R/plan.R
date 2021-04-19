plan <- drake_plan(
    data_result = {
        file_in("kcmkc-results.sqlite")
        table_result()
    },
    plot_tradeoff = target(
        do_plot_tradeoff(filter(data_result, rank == rank_value)),
        transform = cross(
            rank_value = c(10, 100)
        )
    ),
    plot_time = target(
        do_plot_time(filter(data_result, rank == rank_value), coreset_only = F),
        transform = cross(
            rank_value = c(10, 100)
        )
    ),
    plot_time_coreset = target(
        do_plot_time(filter(data_result, rank == rank_value), coreset_only = T),
        transform = cross(
            rank_value = c(10, 100)
        )
    ),
    plot_param_influence = do_plot_param_influence(data_result),
    notes = rmarkdown::render(
        knitr_in("R/notes.Rmd"),
        output_file = file_out("notes.html"),
        output_dir = "R",
        quiet = TRUE
    )
)
