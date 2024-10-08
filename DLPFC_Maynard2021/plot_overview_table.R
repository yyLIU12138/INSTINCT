#' Visualizing scores in an overview table
#'
#' \code{scIB_knit_table} returns a ggplot2 object with a ranking of the integration methods  
#' and their relative scores over a set of metrics.
#'
#' @param data A data frame with methods by row, metrics by columns. First column must be method's names. Score values
#' should be scaled between [0,1].
#' @param column_info A data frame describing the columns of `data`. This data frame should contain the following columns:
#'   * `id` (`character`): The corresponding column name in `data`.
#'   * `color_group` (`character`): The group name of the corresponding palette of the column.
#'   * `geom` (`character`): The geom of the column. Must be one of: `circle`, `bar`, or `text`.
#'   * `width`: Custom width for this column (default: 1).
#'   * `overlay`: Whether to overlay this column over the previous column. If so, the width of that column will be inherited.
#' @param row_info A data frame describing the rows of `data`. This data should contain the following columns:`
#'   * `id` (`character`): The corresponding row name in `data`.
#'   * `group` (`character`): The group of the row. If all are `NA`, the rows will not be split up into groups.
#' @param palettes A list of palettes to be used in the respective `color_group`.
#'
#'
#' @importFrom ggforce geom_arc_bar geom_circle geom_arc
#' @importFrom cowplot theme_nothing
#'

library(scales)
library(ggimage)
library(cowplot)
library(RColorBrewer)
library(dplyr)
library(tibble)


scIB_knit_table <- function(
  data,
  dataset,
  column_info,
  row_info,
  palettes,
  task = FALSE,
  usability = FALSE,
  atac = FALSE,
  atac_best = FALSE
) {
  # no point in making these into parameters
  row_height <- 1.1  #1.1
  row_space <- 0.3 #0.1
  row_bigspace <- .5
  col_width <- 1.1
  col_space <- .2
  col_bigspace <- .5
  segment_data <- NULL

  # DETERMINE ROW POSITIONS
  if (!"group" %in% colnames(row_info) || all(is.na(row_info$group))) {
    row_info$group <- ""
    row_groups <- tibble(group = "")
    plot_row_annotation <- FALSE
  } else {
    plot_row_annotation <- TRUE
  }

  row_pos <-
    row_info %>%
    group_by(group) %>%
    mutate(group_i = row_number()) %>%
    ungroup() %>%
    mutate(
      row_i = row_number(),
      colour_background = group_i %% 2 == 1,
      do_spacing = c(FALSE, diff(as.integer(factor(group))) != 0),
      ysep = ifelse(do_spacing, row_height + 2 * row_space, row_space),
      y = - (row_i * row_height + cumsum(ysep)),
      ymin = y - row_height / 2,
      ymax = y + row_height / 2
    )

  # DETERMINE COLUMN POSITIONS
  if (!"group" %in% colnames(column_info) || all(is.na(column_info$group))) {
    column_info$group <- ""
    plot_column_annotation <- FALSE
  } else {
    plot_column_annotation <- TRUE
  }
  
  column_info <-
    column_info %>%
    add_column_if_missing(width = col_width, overlay = FALSE)
  

  column_pos <-
    column_info %>%
    mutate(
      do_spacing = c(FALSE, diff(as.integer(factor(group))) != 0),
      xsep = case_when(
        overlay ~ c(0, -head(width, -1)),
        do_spacing ~ col_bigspace,
        TRUE ~ col_space
      ),
      xwidth = case_when(
        overlay & width < 0 ~ width - xsep,
        overlay ~ -xsep,
        TRUE ~ width
      ),
      xmax = cumsum(xwidth + xsep),
      xmin = xmax - xwidth,
      x = xmin + xwidth / 2
    )

  ##########################
  #### CREATE GEOM DATA ####
  ##########################
  
  # gather circle data
  ind_circle <- which(column_info$geom == "circle")
  if(length(ind_circle) > 0){
    dat_mat <- as.matrix(data[, ind_circle])
    col_palette <- data.frame(metric = colnames(dat_mat), 
                              group = column_info[match(colnames(dat_mat), column_info$id), "group"])
    
    col_palette$name_palette <- lapply(col_palette$group, function(x) palettes[[as.character(x)]])

    circle_data <- data.frame(label = unlist(lapply(colnames(dat_mat),
                                                    function(x) rep(x, nrow(dat_mat)))),
                              x0 = unlist(lapply(column_pos$x[ind_circle],
                                                 function(x) rep(x, nrow(dat_mat)))),
                              y0 = rep(row_pos$y, ncol(dat_mat)),
                              r = row_height / 2 * as.vector(sqrt(pmax(dat_mat, 0.0001)))
                              )

    for (l in unique(circle_data$label)) {
      ind_l <- which(circle_data$label == l)
      circle_data[ind_l, "r"] <- rescale(circle_data[ind_l, "r"],
        to = c(0.1, 1) * row_height / 2,
        from = range(circle_data[ind_l, "r"], na.rm = T))
    }

    colors <- NULL

    for(i in 1:ncol(dat_mat)){
      palette <- colorRampPalette(rev(brewer.pal(9, col_palette$name_palette[[i]])))(nrow(data)-sum(is.na(dat_mat[,i])))
      colors <- c(colors, palette[rank(dat_mat[,i], ties.method = "average", na.last = "keep")])
    }

    circle_data$colors <- colors
  }

  
  # gather bar data
  ind_bar <- which(column_info$geom == "bar")
  dat_mat <- as.matrix(data[, ind_bar])
  
  col_palette <- data.frame(metric = colnames(dat_mat),
                            group = column_info[match(colnames(dat_mat), column_info$id), "group"])
  
  col_palette$name_palette <- lapply(col_palette$group, function(x) palettes[[as.character(x)]])
  
  
  rect_data <- data.frame(label = unlist(lapply(colnames(dat_mat), 
                                                function(x) rep(x, nrow(dat_mat)))),
                          method = rep(row_info$id, ncol(dat_mat)),
                          value = as.vector(dat_mat),
                          xmin = unlist(lapply(column_pos[ind_bar, "xmin"], 
                                               function(x) rep(x, nrow(dat_mat)))),
                          xmax = unlist(lapply(column_pos[ind_bar, "xmax"], 
                                               function(x) rep(x, nrow(dat_mat)))),
                          ymin = rep(row_pos$ymin, ncol(dat_mat)),
                          ymax = rep(row_pos$ymax, ncol(dat_mat)),
                          xwidth = unlist(lapply(column_pos[ind_bar, "xwidth"], 
                                                 function(x) rep(x, nrow(dat_mat))))
  )
  rect_data <- rect_data %>%
    add_column_if_missing(hjust = 0) %>%
    mutate(
      xmin = xmin + (1 - value) * xwidth * hjust,
      xmax = xmax - (1 - value) * xwidth * (1 - hjust)
    )
  
  colors <- NULL
  for(i in 1:ncol(dat_mat)){
    palette <- colorRampPalette(rev(brewer.pal(9, col_palette$name_palette[[i]])))(nrow(data)-sum(is.na(dat_mat[,i])))
    colors <- c(colors, palette[rank(dat_mat[,i], ties.method = "average", na.last = "keep")])
  }
  
  rect_data$colors <- colors

  
  # gather bar guides data
  ind_bar <- which(column_info$geom == "bar")
  dat_mat <- as.matrix(data[, ind_bar])

  xmin <- unlist(lapply(column_pos[ind_bar, "xmin"], function(x) rep(x, nrow(dat_mat))))
  xmax <- unlist(lapply(column_pos[ind_bar, "xmax"],  function(x) rep(x, nrow(dat_mat))))

  barguides_data <- data.frame(x = c(xmin, xmax),
                               xend = c(xmin, xmax),
                               y = rep(row_pos$ymin, ncol(dat_mat)),
                               yend = rep(row_pos$ymax, ncol(dat_mat)),
                               size = .5)

  barguides_data <- barguides_data %>%
                    add_column_if_missing(hjust = 0) %>%
                    mutate(palette = NA, value = NA)
                    
  segment_data <- bind_rows(segment_data, barguides_data %>%
                            mutate(colour = "black", size = .5, linetype = "dashed"))

  
  # gather text data
  ind_text <- which(column_info$geom == "text")
  dat_mat <- as.matrix(data[, ind_text])

  column_pos[ind_text, "xmin"] <- column_pos[ind_text, "xmin"] - 2
  column_pos[ind_text, "xmax"] <- column_pos[ind_text, "xmax"] - 3.5
  text_data <- data.frame(label_value = as.vector(dat_mat),
                          # group = rep(colnames(dat_mat), each = nrow(dat_mat)),
                          group = dat_mat,
                          xmin = unlist(lapply(column_pos[ind_text, "xmin"], 
                                               function(x) rep(x, nrow(dat_mat)))),
                          xmax = unlist(lapply(column_pos[ind_text, "xmax"], 
                                              function(x) rep(x, nrow(dat_mat)))),
                          ymin = rep(row_pos$ymin, ncol(dat_mat)),
                          ymax = rep(row_pos$ymax, ncol(dat_mat)),
                          size = 4, fontface = "plain", stringsAsFactors = F)
  
  text_data$colors <- "black"
  text_data[text_data$label_value == "HVG", "colors"] <- "darkgreen"
  text_data[text_data$label_value == "FULL", "colors"] <- "grey30"
  
  # # replace scaled/unscaled with +/-
  # text_data$label_value <- mapvalues(text_data$label_value, from = c("scaled", "unscaled"), 
  #                                    to = c("+", "-"))
  
  text_data[text_data$label_value == "+" | text_data$label_value == "-", "size"] <- 5
  text_data[text_data$label_value == "+" | text_data$label_value == "-", "fontface"] <- "bold"
  
  text_data[text_data$label_value == "genes" | text_data$label_value == "peaks" | text_data$label_value == "windows", "size"] <- 3

  
  # ADD top3 ranking for each bar column
  if(usability || atac_best){
    cols_bar <- unique(rect_data$label)
    cols_bar <- as.character(cols_bar[!is.na(cols_bar)])
    for(c in cols_bar){
      rect_tmp <- rect_data[rect_data$label == c,]
      rect_tmp <- add_column(rect_tmp, "label_value" = as.character(rank(-rect_tmp$value, ties.method = "min")))
      rect_tmp <- rect_tmp[rect_tmp$label_value %in% c("1", "2", "3"), c("label_value", "xmin", "xmax", "ymin", "ymax")]
      rect_tmp <- add_column(rect_tmp, "size" = 2.5, .after = "ymax")
      rect_tmp <- add_column(rect_tmp, "colors" = "black", .after = "size")
      rect_tmp <- add_column(rect_tmp, "fontface" = "plain", .after = "colors")
      rect_tmp <- add_column(rect_tmp, "group" = "top3", .after = "fontface")
      text_data <- bind_rows(text_data, rect_tmp)
    }
  }
  
  
  
  # ADD COLUMN NAMES
  df <- column_pos %>% filter(id != "Method") %>% filter(id != "Ranking")
 
  if (nrow(df) > 0) {
    segment_data <- segment_data %>% bind_rows(
      df %>% transmute(x = x, xend = x, y = -6.3, yend = -6.5, size = .5)
    )
    text_data <-
      bind_rows(
        text_data,
        df %>% transmute(
          xmin = x, xmax = x, ymin = -6.3, ymax = -6.8,
          angle = 30, vjust = 0.5, hjust = 1,
          label_value = id, 
          size = 3.5
        )
      )
  }
 
  
  # GENERATE ROW ANNOTATION
  if (plot_row_annotation) {
    row_annotation <-
      row_pos %>% 
      select(group, ymin, ymax) %>%
      group_by(group) %>%
      summarise(
        ymin = min(ymin),
        ymax = max(ymax),
        y = (ymin + ymax) / 2
      ) %>%
      ungroup() %>%
      mutate(xmin = -.5, xmax = 5) %>%
      filter(!is.na(group), group != "")
    
    text_data <- text_data %>% bind_rows(
      row_annotation %>%
        transmute(xmin, xmax, ymin = ymax + row_space, label_value = group %>% gsub("\n", " ", .), 
                  hjust = 0, vjust = .5, fontface = "bold", size = 4) %>%
        mutate(ymax = ymin + row_height)
    )
  }
  
  # gather image data
  ind_img <- which(column_info$geom == "image")
  if(length(ind_img) > 0){
    dat_mat <- as.matrix(data[, ind_img])

    if (!task) {
      imgs <- c("./img/matrix.png", "./img/embedding.png", "./img/graph.png")
      img_map <-  mapvalues(dat_mat, from = c("graph", "embed", "gene"),
          to = imgs)
    } else {
      imgs <- c(
        "./img/full.png", "./img/paired+full.png", "./img/single+full.png", "./img/paired_abc.png",
        "./img/paired_ab.png", "./img/paired_ac.png", "./img/paired_bc.png", "./img/single.png",
        "./img/single_atac.png", "./img/single_rna.png", "./img/single_adt.png",
        "./img/paired_a.png", "./img/paired_b.png", "./img/paired_c.png"
      )
      img_map <-  mapvalues(dat_mat, from = c(
        "full", "paired_full", "single_full", "paired_abc",
        "paired_ab", "paired_ac", "paired_bc", "single",
        "single_atac", "single_rna", "single_adt",
        "paired_a", "paired_b", "paired_c"
        ), to = imgs)
    }
    
    image_data <- data.frame(x = unlist(lapply(column_pos$x[ind_img],
                                                function(x) rep(x, nrow(dat_mat)))),
                             y = rep(row_pos$y, ncol(dat_mat)),
                             image = img_map,
                             stringsAsFactors = FALSE
                             )
  }
  suppressWarnings({
    minimum_x <- min(column_pos$xmin, segment_data$x, segment_data$xend, 
                     text_data$xmin, na.rm = TRUE)
    maximum_x <- max(column_pos$xmax, segment_data$x, segment_data$xend, 
                     text_data$xmax, na.rm = TRUE)
    minimum_y <- min(row_pos$ymin, segment_data$y, segment_data$yend,  
                     text_data$ymin, na.rm = TRUE)
    maximum_y <- max(row_pos$ymax, segment_data$y, segment_data$yend, 
                     text_data$ymax, na.rm = TRUE)
  })
  
  ####################################
  ###   CREATE HARDCODED LEGENDS   ###
  ####################################
  
  x_min_output <- minimum_x + 0.5
  # x_min_scaling <- minimum_x + 5.5
  x_min_ranking <- ifelse(atac, minimum_x + 8.5, minimum_x + 15)
  x_min_score <-  ifelse(atac, minimum_x + 14, minimum_x + 21.5)
  
  leg_max_y <- minimum_y - .5
  
#   # Create legend for Output
#   leg_min_x <- x_min_output
# 
#   output_title_data <- data.frame(xmin = leg_min_x,
#                                 xmax = leg_min_x + 2, 
#                                 ymin = leg_max_y - 1, 
#                                 ymax = leg_max_y, 
#                                 label_value = "Task", 
#                                 hjust = 0, vjust = 0, 
#                                 fontface = "bold",
#                                 size = 3)
#   
#   img_min_x <- c(rep(leg_min_x + 0.5, 8), rep(leg_min_x + 7.5, 6))
# #   output_img <- data.frame(
# #     x = img_min_x,
# #     y = c(leg_max_y-2, leg_max_y-3.2, leg_max_y-4.4, leg_max_y-5.6, leg_max_y-6.8, leg_max_y-8.0, leg_max_y-9.2, leg_max_y-10.4,
# #           leg_max_y-2, leg_max_y-3.2, leg_max_y-4.4, leg_max_y-5.6, leg_max_y-6.8, leg_max_y-8.0),
# #     image = imgs)
# 
#   output_text <- data.frame(
#     xmin = img_min_x + 0.8,
#     xmax = img_min_x + 0.8 + 5,
#     ymin = c(leg_max_y-2, leg_max_y-3.2, leg_max_y-4.4, leg_max_y-5.6, leg_max_y-6.8, leg_max_y-8.0, leg_max_y-9.2, leg_max_y-10.4,
#              leg_max_y-2, leg_max_y-3.2, leg_max_y-4.4, leg_max_y-5.6, leg_max_y-6.8, leg_max_y-8.0) - 0.2,
#     ymax = c(leg_max_y-2, leg_max_y-3.2, leg_max_y-4.4, leg_max_y-5.6, leg_max_y-6.8, leg_max_y-8.0, leg_max_y-9.2, leg_max_y-10.4,
#              leg_max_y-2, leg_max_y-3.2, leg_max_y-4.4, leg_max_y-5.6, leg_max_y-6.8, leg_max_y-8.0) - 0.2,
#     label_value = dataset,
#     hjust = 0, vjust = 0,
#     fontface = "plain",
#     size = 3)
# 
#   
#   text_data <- bind_rows(text_data, output_text, output_title_data)
# #   image_data <- bind_rows(image_data, output_img)
# 
#   # # Create legend for scaling
#   # if(!atac && !atac_best){
#   # leg_min_x <- x_min_scaling
#   # scaling_title_data <- data.frame(xmin = leg_min_x, 
#   #                                 xmax = leg_min_x+ 2, 
#   #                                 ymin = leg_max_y - 1, 
#   #                                 ymax = leg_max_y, 
#   #                                 label_value = "Scaling", 
#   #                                 hjust = 0, vjust = 0, 
#   #                                 fontface = "bold",
#   #                                 size = 3)
#   
#   # scaling_text <- data.frame(xmin = c(leg_min_x, leg_min_x+1), 
#   #                           xmax = c(leg_min_x+0.5, leg_min_x+3), 
#   #                           ymin = c(rep(leg_max_y-2,2), rep(leg_max_y-3,2)), 
#   #                           ymax = c(rep(leg_max_y-1,2), rep(leg_max_y-2,2)), 
#   #                           label_value = c("+", ": scaled", "-", ": unscaled"), 
#   #                           hjust = 0, vjust = 0, 
#   #                           fontface = c("bold","plain", "bold", "plain"),
#   #                           size = c(5,3,5,3))
#   
#   # text_data <- bind_rows(text_data, scaling_title_data, scaling_text)
#   # }
  
  # CREATE LEGEND for ranking colors
  leg_min_x <- x_min_ranking
  rank_groups <- as.character(column_info[column_info$geom == "bar", "group"])
  
  if(usability){
    rank_minimum_x <- list("RNA" = leg_min_x, 
                           "Simulation" = leg_min_x+1, 
                           "Usability" = leg_min_x+2,
                           "Scalability" = leg_min_x+3)
    leg_max_x <- leg_min_x+3
  } else if(atac_best){
    rank_minimum_x <- list("ATAC_windows" = leg_min_x, 
                           "ATAC_peaks" = leg_min_x+1, 
                           "ATAC_genes" = leg_min_x+2)
    leg_max_x <- leg_min_x+2
  } else{
    rank_minimum_x <- list()
    j <- 0
    for (i in seq_along(column_info$geom)) {
        if (column_info$geom[i] == "bar") {
            rank_minimum_x[[column_info$group[i]]] <- leg_min_x + j
            j <- j + 1
        }
    }
    leg_max_x <- leg_min_x + j
  }
  
  rank_title_data <- data.frame(xmin = leg_min_x, 
                                xmax = leg_min_x+ 2, 
                                ymin = leg_max_y - 2.5, 
                                ymax = leg_max_y - 1.5, 
                                label_value = "Ranking", 
                                hjust = 0, vjust = 0, 
                                fontface = "bold")

  for(rg in rank_groups){
    rank_palette <- colorRampPalette(rev(brewer.pal(9, palettes[[rg]])))(4)
    
    rank_data <- data.frame(xmin = rank_minimum_x[[rg]],
                            xmax = rank_minimum_x[[rg]] + .8,
                            ymin = seq(leg_max_y-5, leg_max_y - 3.5, by = .5),
                            ymax = seq(leg_max_y-4.5, leg_max_y -3, by = .5),
                            border = TRUE,
                            colors = rank_palette
    )
    rect_data <- bind_rows(rect_data, rank_data)
  }
  

  # create arrow for ranking
  arrow_data <- data.frame(x = leg_max_x + 0.5, 
                           xend = leg_max_x + 0.5,
                           y = leg_max_y-5, 
                           yend = leg_max_y -3)
  

  # add text next to the arrow
  arrow_text <- data.frame(xmin = leg_max_x + 1,
                           xmax = leg_max_x + 1.5,
                           ymin = c(leg_max_y - 3.5, leg_max_y - 5),
                           ymax = c(leg_max_y - 3, leg_max_y - 4.5),
                           label_value = c("1", as.character(nrow(data))),
                           hjust = 0, vjust = 0, size = 2.5)

  
  text_data <- bind_rows(text_data, rank_title_data, arrow_text)

  # CREATE LEGEND for circle scores
  # circle legend
  if(!usability && !atac_best){
    cir_minimum_x <- x_min_score
    
    cir_legend_size <- 1
    cir_legend_space <- .1
  
    cir_legend_dat <-
      data.frame(
        value = seq(0, 1, by = .2),
        r = row_height/2*seq(0, 1, by = .2)
      )
    cir_legend_dat$r <- rescale(cir_legend_dat$r, to = c(0.05, 0.55), from = range(cir_legend_dat$r, na.rm = T))
  
    x0 <- vector("integer", nrow(cir_legend_dat))
    for(i in 1:length(x0)){
      if(i == 1){
        x0[i] <- cir_minimum_x + cir_legend_space + cir_legend_dat$r[i]
      }
      else {
        x0[i] <- x0[i-1] + cir_legend_dat$r[i-1] + cir_legend_space + cir_legend_dat$r[i]
      }
    }
  
    cir_legend_dat$x0 <- x0
    cir_legend_min_y <- leg_max_y - 4
    cir_legend_dat$y0 <- cir_legend_min_y + cir_legend_dat$r
  
    cir_legend_dat$colors <- NULL
    cir_maximum_x <- max(cir_legend_dat$x0)
  
    cir_title_data <- tibble(xmin = cir_minimum_x, 
                                 xmax = cir_maximum_x, 
                                 ymin = leg_max_y - 2.5, 
                                 ymax = leg_max_y - 1.5,
                                 label_value = "Score", 
                                 hjust = 0, vjust = 0, fontface = "bold")
    
    cir_value_data <- data.frame(xmin = cir_legend_dat$x0 - cir_legend_dat$r,
                                 xmax = cir_legend_dat$x0 + cir_legend_dat$r,
                                 ymin = cir_legend_min_y - 0.5,
                                 ymax = cir_legend_min_y + 2.5,
                                 hjust = .5, vjust = 0, size = 2.5,
                                 label_value = ifelse(cir_legend_dat$value %in% c(0, 1), 
                                                      paste0(cir_legend_dat$value*100, "%"), ""))
    
    circle_data <- bind_rows(circle_data, cir_legend_dat)
    text_data <- bind_rows(text_data, cir_title_data, cir_value_data)
  
  
    
  }
  
  minimum_y <- min(minimum_y, min(text_data$ymin, na.rm = TRUE))
  
  ########################
  ##### COMPOSE PLOT #####
  ########################
  
  g <-
    ggplot() +
    coord_equal(expand = FALSE) +
    scale_alpha_identity() +
    scale_colour_identity() +
    scale_fill_identity() +
    scale_size_identity() +
    scale_linetype_identity() +
    cowplot::theme_nothing()
  
  # PLOT ROW BACKGROUNDS
  df <- row_pos %>% filter(colour_background)
  if (nrow(df) > 0) {
    g <- g + geom_rect(aes(xmin = min(column_pos$xmin)-.25, xmax = max(column_pos$xmax)+.25, ymin = ymin - (row_space / 2), ymax = ymax + (row_space / 2)), df, fill = "#DDDDDD")
  } 
  
  
  
  # PLOT CIRCLES
  if (length(ind_circle) > 0) {
    g <- g + ggforce::geom_circle(aes(x0 = x0, y0 = y0, fill= colors, r = r), circle_data, size=.25)
  }
  
  
  # PLOT RECTANGLES
  if (nrow(rect_data) > 0) {
    # add defaults for optional values
    rect_data <- rect_data %>%
      add_column_if_missing(alpha = 1, border = TRUE, border_colour = "black") %>%
      mutate(border_colour = ifelse(border, border_colour, NA))
    
    g <- g + geom_rect(aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = colors, colour = border_colour, alpha = alpha), rect_data, size = .25)
  }
  
  
  # PLOT TEXT
  if (nrow(text_data) > 0) {
    # add defaults for optional values
    text_data <- text_data %>%
      add_column_if_missing(
        hjust = .5,
        vjust = .5,
        size = 3,
        fontface = "plain",
        colors = "#0a0a0a",
        lineheight = 1,
        angle = 0
      ) %>%
      mutate(
        angle2 = angle / 360 * 2 * pi,
        cosa = cos(angle2) %>% round(2),
        sina = sin(angle2) %>% round(2),
        alphax = ifelse(cosa < 0, 1 - hjust, hjust) * abs(cosa) + ifelse(sina > 0, 1 - vjust, vjust) * abs(sina),
        alphay = ifelse(sina < 0, 1 - hjust, hjust) * abs(sina) + ifelse(cosa < 0, 1 - vjust, vjust) * abs(cosa),
        x = (1 - alphax) * xmin + alphax * xmax,
        y = (1 - alphay) * ymin + alphay * ymax
      ) %>%
      filter(label_value != "")
    # Set fontface for legend bold
    text_data[text_data$label_value == "Ranking", "fontface"] <- "bold"
    # # Set fontface for ranking numbers bold
    # if(usability || atac_best){
    # text_data[1:nrow(data), "fontface"] <- "bold"
    # }
    # subset text_data to left-aligned rows
    text_data_left <- text_data[which(text_data$group != "NA" | text_data$group == "top3"), ]
    text_data <- text_data[-which(text_data$group != "NA" | text_data$group == "top3"), ]

    g <- g + geom_text(aes(x = x, y = y, label = label_value, colour = colors, hjust = hjust, vjust = vjust, size = size, fontface = fontface, angle = angle), data = text_data)
    
    # text_data_left[text_data_left$group == "Method", "x"] <- text_data_left[text_data_left$group == "Method", "x"] - 3
    # if(usability || atac_best){
    # text_data_left[text_data_left$group == "top3", "x"] <- text_data_left[text_data_left$group == "top3", "xmin"] + .3
    # text_data_left[text_data_left$group == "Method", "x"] <- text_data_left[text_data_left$group == "Method", "x"] + .5
    # }
    g <- g + geom_text(aes(x = x, y = y, label = label_value, colour = colors, hjust = "left", vjust = vjust, size = size, fontface = fontface, angle = angle), data = text_data_left)
  }
  
  
  
  # PLOT SEGMENTS
  if (nrow(segment_data) > 0) {
    # add defaults for optional values
    segment_data <- segment_data %>% add_column_if_missing(size = .5, colour = "black", linetype = "solid")
    
    g <- g + geom_segment(aes(x = x, xend = xend, y = y, yend = yend, size = size, colour = colour, linetype = linetype), segment_data)
  }
  
  # PLOT ARROW RANKING
  if (nrow(arrow_data) > 0) {
    # add defaults for optional values
    arrow_data <- arrow_data %>% add_column_if_missing(size = .5, colour = "black", linetype = "solid")
    
    g <- g + geom_segment(aes(x = x, xend = xend, y = y, yend = yend, size = size, colour = colour, linetype = linetype), arrow_data, arrow = arrow(length = unit(0.1, "cm")), lineend = "round", linejoin = "bevel")
  }
  
  # PLOT IMAGES
  if(length(ind_img) > 0){
    for(r in 1:nrow(image_data)){
      g <- g + cowplot::draw_image(image = image_data$image[r], x = image_data[r, "x"]-.5, y = image_data[r, "y"]-.5)
    }
    
  }
  
  # ADD SIZE
  # reserve a bit more room for text that wants to go outside the frame
  minimum_x <- minimum_x - 2
  maximum_x <- maximum_x + 5
  minimum_y <- minimum_y - 2
  maximum_y <- maximum_y + 4
  
  g$width <- maximum_x - minimum_x
  g$height <- maximum_y - minimum_y
  
  g <- g + expand_limits(x = c(minimum_x, maximum_x), y = c(minimum_y, maximum_y))
  
  return(g)
  
}



add_column_if_missing <- function(df, ...) {
  column_values <- list(...)
  for (column_name in names(column_values)) {
    default_val <- rep(column_values[[column_name]], nrow(df))
    
    if (column_name %in% colnames(df)) {
      df[[column_name]] <- ifelse(is.na(df[[column_name]]), default_val, df[[column_name]])
    } else {
      df[[column_name]] <- default_val
    }
  }
  df
}


save_dir <- 'C:/University/FINAL/results/DLPFC_Maynard2021/'
group_idx <- 0


## column_info
column_id <- c('Name', 'ARI', 'AMI', 'NMI', 'FMI', 'Comp', 'Homo', 'Overall score 1',
               'mAP', 'Spot-type ASW', 'Overall score 2', 'Batch ASW', 'Batch PCR', 
               'kBET', 'Graph connectivity', 'Overall score 3', 'Overall score')

column_color_group <- c('Methods', rep('Clustering Performance', 7), 
                        rep('Representation Quality', 3), rep('Batch Correction', 5), 'Overall')

column_geom <- c('text', rep('circle', 6), 'bar', rep('circle', 2), 'bar', rep('circle', 4), 'bar', 'bar')

column_width <- rep(1.3, length(id))

column_overlay <- rep(FALSE, length(id))

column_info <- data.frame(id = column_id, color_group = column_color_group, geom = column_geom, 
                          width = column_width, overlay = column_overlay, group = column_color_group)


## row_info
row_id <- c('INSTINCT', 'SEDR', 'STAligner', 'GraphST')

row_group <- rep(NA, length(row_id))

row_info <- data.frame(id = row_id, group = row_group)


# palette
palette_list <- list(
    'Methods' = 'Greys',
    'Clustering Performance' = "Oranges",
    'Representation Quality' = "Greens",
    'Batch Correction' = "Blues",
    'Overall' = "Reds"
)


filename <- paste(save_dir, 'comparison/group', group_idx, '_methods_mean_separate_scores.txt', sep = '')
data_mtx <- read.table(filename, header = FALSE)

row_names <- c('INSTINCT', 'SEDR', 'STAligner', 'GraphST')

data_df <- cbind(row_names, data_mtx)
colnames(data_df) <- column_id

g1 = scIB_knit_table(
     data = data_df,
     dataset = NA,
     column_info = column_info,
     row_info = row_info,
     palettes = palette_list,
     task = FALSE,
     usability = FALSE,
     atac = FALSE,
     atac_best = FALSE
)


filename <- paste(save_dir, 'comparison/group', group_idx, '_methods_median_separate_scores.txt', sep = '')
data_mtx <- read.table(filename, header = FALSE)

row_names <- c('INSTINCT', 'SEDR', 'STAligner', 'GraphST')

data_df <- cbind(row_names, data_mtx)
colnames(data_df) <- column_id

g2 = scIB_knit_table(
     data = data_df,
     dataset = NA,
     column_info = column_info,
     row_info = row_info,
     palettes = palette_list,
     task = FALSE,
     usability = FALSE,
     atac = FALSE,
     atac_best = FALSE
)


ggsave(paste0(save_dir, 'comparison/methods_mean_group', group_idx, '_overview_table.pdf'), g1, width = 10, height = 7)
ggsave(paste0(save_dir, 'comparison/methods_median_group', group_idx, '_overview_table.pdf'), g2, width = 10, height = 7)

