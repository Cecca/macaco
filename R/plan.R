plan <- drake_plan(
    data_result = {
        file_in("kcmkc-results.sqlite")
        table_result()
    },
    plot_tradeoff = do_plot_tradeoff(data_result),
    # fig_tradeoff = {
    #     ggsave(
    #         filename = file_out("imgs/tradeoff.pdf"),
    #         plot = plot_tradeoff,
    #         width = 8,
    #         height = 6
    #     )
    # },
    plot_param_influence = do_plot_param_influence(data_result),
    fig_param_influence = {
        ggsave(
            filename = file_out("imgs/param_influence.pdf"),
            plot = plot_param_influence,
            width = 6,
            height = 6
        )
    },
    notes = rmarkdown::render(
        knitr_in("R/notes.Rmd"),
        output_file = file_out("notes.html"),
        output_dir = "R",
        quiet = TRUE
    )
)