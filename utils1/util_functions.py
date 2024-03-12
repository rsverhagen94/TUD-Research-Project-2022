import sys, random, enum, ast, time, threading, os, math, contextlib
from rpy2 import robjects
    
def R_to_Py_plot_priority(people, smoke, duration, location, image_name):
    r_script = (f'''
                data <- read_excel("/home/ruben/Downloads/moral sensitivity survey data 4.xlsx")
                data$situation <- as.factor(data$situation)
                data$location <- as.factor(data$location)
                data$smoke <- as.factor(data$smoke)
                data_subset <- subset(data, data$situation=="3"|data$situation=="6")
                data_subset <- data_subset[data_subset$smoke != "pushing out",]
                data_subset$people <- as.numeric(data_subset$people)
                fit <- lm(sensitivity ~ people + duration + smoke + location, data = data_subset[-c(244,242,211,162,96,92,29),])
                pred_data <- subset(data_subset[-c(244,242,211,162,96,92,29),], select = c("people", "duration", "smoke", "location", "sensitivity"))
                pred_data$smoke <- factor(pred_data$smoke, levels = c("fast", "normal", "slow"))
                explainer <- shapr(pred_data, fit)
                p <- mean(pred_data$sensitivity)
                new_data <- data.frame(people = c({people}),
                                    duration = c({duration}),
                                    smoke = c("{smoke}"),
                                    location = c("{location}"))
                new_data$smoke <- factor(new_data$smoke, levels = c("fast", "normal", "slow"))
                new_data$location <- factor(new_data$location, levels = c("known", "unknown"))
                new_pred <- predict(fit, new_data)
                explanation_cat <- shapr::explain(new_data, approach = "ctree", explainer = explainer, prediction_zero = p)
                shapley_values <- explanation_cat[["dt"]][,2:5]
                standardized_values <- shapley_values / sum(abs(shapley_values))
                explanation_cat[["dt"]][,2:5] <- standardized_values
                pl <- plot(explanation_cat, digits = 1, plot_phi0 = FALSE) 
                pl[["data"]]$header <- paste("predicted sensitivity = ", round(new_pred, 1), sep = " ")
                data_plot <- pl[["data"]]
                min <- 'min.'
                loc <- NA
                if ("{location}" == 'known') {{
                    loc <- 'found'
                }}
                if ("{location}" == 'unknown') {{
                    loc <- '?'
                }}
                labels <- c(duration = paste("<img src='/home/ruben/xai4mhc/Icons/duration_fire_black.png' width='31' /><br>\n", new_data$duration, min), 
                smoke = paste("<img src='/home/ruben/xai4mhc/Icons/smoke_speed_black.png' width='53' /><br>\n", new_data$smoke), 
                location = paste("<img src='/home/ruben/xai4mhc/Icons/location_fire_black.png' width='35' /><br>\n", loc), 
                people = paste("<img src='/home/ruben/xai4mhc/Icons/victims.png' width='19' /><br>\n", new_data$people))
                data_plot$variable <- reorder(data_plot$variable, -abs(data_plot$phi))
                pl <- ggplot(data_plot, aes(x = variable, y = phi, fill = ifelse(phi >= 0, "positive", "negative"))) +
                geom_bar(stat = "identity") +
                scale_x_discrete(name = NULL, labels = labels) +
                theme(axis.text.x = ggtext::element_markdown()) + # Removed color and size attributes here
                theme(text = element_text(size = 12, family = "sans-serif"), # Changed size to 12, which is roughly 1rem
                        plot.title = element_text(hjust = 0.5, size = 12, color = "black", face = "bold", margin = margin(b = 5)),
                        plot.caption = element_text(size = 12, margin = margin(t = 25), color = "black"),
                        panel.background = element_blank(),
                        axis.text = element_text(size = 12, color = "black"),
                        axis.text.y = element_text(color = "black", margin = margin(t = 5)),
                        axis.line = element_line(color = "black"),
                        axis.title = element_text(size = 12),
                        axis.title.y = element_text(color = "black", margin = margin(r = 10), hjust = 0.5),
                        axis.title.x = element_text(color = "black", margin = margin(t = 5), hjust = 0.5),
                        panel.grid.major = element_line(color = "#DAE1E7"),
                        panel.grid.major.x = element_blank()) +
                theme(legend.background = element_rect(fill = "white", color = "white"),
                        legend.key = element_rect(fill = "white", color = "white"),
                        legend.text = element_text(size = 12),
                        legend.position = "none",
                        legend.title = element_text(size = 12, face = "plain")) +
                labs(y = "Relative feature contribution", fill = "") +
                scale_y_continuous(breaks = seq(-1, 1, by = 0.5), limits = c(-1, 1), expand = c(0.0, 0.0)) +
                scale_fill_manual(values = c("positive" = "#3E6F9F", "negative" = "#B0D7F0"), breaks = c("positive", "negative")) +
                geom_hline(yintercept = 0, color = "black") +
                theme(axis.text = element_text(color = "black"),
                        axis.ticks = element_line(color = "black"))
                dpi_web <- 300
                width_pixels <- 1500
                height_pixels <- 1500
                width_inches_web <- width_pixels / dpi_web
                height_inches_web <- height_pixels / dpi_web
                ggsave(filename="{image_name}", plot=pl, width=width_inches_web, height=height_inches_web, dpi=dpi_web)
                ''')
    with open(os.devnull, 'w') as nullfile:
        with contextlib.redirect_stdout(nullfile), contextlib.redirect_stderr(nullfile):
            robjects.r(r_script)
    sensitivity = robjects.r['new_pred'][0]
    return round(sensitivity, 1)

def R_to_Py_plot_tactic(people, location, duration, resistance, image_name):
    r_script = (f'''
                data <- read_excel("/home/ruben/Downloads/moral sensitivity survey data 4.xlsx")
                data$situation <- as.factor(data$situation)
                data$location <- as.factor(data$location)
                data_subset <- subset(data, data$situation=="5"|data$situation=="7")
                data_subset$people[data_subset$people == "0"] <- "none"
                data_subset$people[data_subset$people == "1"] <- "one"
                data_subset$people[data_subset$people == "10" |data_subset$people == "11" |data_subset$people == "2" |data_subset$people == "3" |data_subset$people == "4" |data_subset$people == "5"] <- "multiple"
                data_subset <- data_subset[data_subset$people != "clear",]
                data_subset$people <- factor(data_subset$people, levels = c("none","unclear","one","multiple"))
                fit <- lm(sensitivity ~ people + duration + resistance + location, data = data_subset[-c(266,244,186,178,126,111,97,44,19),])
                pred_data <- subset(data_subset[-c(266,244,186,178,126,111,97,44,19),], select = c("people", "duration", "resistance", "location", "sensitivity"))
                explainer <- shapr(pred_data, fit)
                p <- mean(pred_data$sensitivity)
                new_data <- data.frame(people = c("{people}"),
                                        duration = c({duration}),
                                        resistance = c({resistance}),
                                        location = c("{location}"))
                new_data$people <- factor(new_data$people, levels = c("none", "unclear", "one", "multiple"))
                new_data$location <- factor(new_data$location, levels = c("known", "unknown"))
                new_pred <- predict(fit, new_data)
                explanation_cat <- shapr::explain(new_data, approach = "ctree", explainer = explainer, prediction_zero = p)
                shapley_values <- explanation_cat[["dt"]][,2:5]
                standardized_values <- shapley_values / sum(abs(shapley_values))
                explanation_cat[["dt"]][,2:5] <- standardized_values
                pl <- plot(explanation_cat, digits = 1, plot_phi0 = FALSE) 
                pl[["data"]]$header <- paste("predicted sensitivity = ", round(new_pred, 1), sep = " ")
                data_plot <- pl[["data"]]
                min <- 'min.'
                loc <- NA
                if ("{location}" == 'known') {{
                    loc <- 'found'
                }}
                if ("{location}" == 'unknown') {{
                    loc <- '?'
                }}
                labels <- c(duration = paste("<img src='/home/ruben/xai4mhc/Icons/duration_fire_black.png' width='31' /><br>\n", new_data$duration, min), 
                resistance = paste("<img src='/home/ruben/xai4mhc/Icons/fire_resistance_black.png' width='38' /><br>\n", new_data$resistance, min), 
                location = paste("<img src='/home/ruben/xai4mhc/Icons/location_fire_black.png' width='35' /><br>\n", loc), 
                people = paste("<img src='/home/ruben/xai4mhc/Icons/victims.png' width='19' /><br>\n", new_data$people))
                data_plot$variable <- reorder(data_plot$variable, -abs(data_plot$phi))
                pl <- ggplot(data_plot, aes(x = variable, y = phi, fill = ifelse(phi >= 0, "positive", "negative"))) +
                geom_bar(stat = "identity") +
                scale_x_discrete(name = NULL, labels = labels) +
                theme(axis.text.x = ggtext::element_markdown()) + # Removed color and size attributes here
                theme(text = element_text(size = 12, family = "sans-serif"), # Changed size to 12, which is roughly 1rem
                        plot.title = element_text(hjust = 0.5, size = 12, color = "black", face = "bold", margin = margin(b = 5)),
                        plot.caption = element_text(size = 12, margin = margin(t = 25), color = "black"),
                        panel.background = element_blank(),
                        axis.text = element_text(size = 12, color = "black"),
                        axis.text.y = element_text(color = "black", margin = margin(t = 5)),
                        axis.line = element_line(color = "black"),
                        axis.title = element_text(size = 12),
                        axis.title.y = element_text(color = "black", margin = margin(r = 10), hjust = 0.5),
                        axis.title.x = element_text(color = "black", margin = margin(t = 5), hjust = 0.5),
                        panel.grid.major = element_line(color = "#DAE1E7"),
                        panel.grid.major.x = element_blank()) +
                theme(legend.background = element_rect(fill = "white", color = "white"),
                        legend.key = element_rect(fill = "white", color = "white"),
                        legend.text = element_text(size = 12),
                        legend.position = "none",
                        legend.title = element_text(size = 12, face = "plain")) +
                labs(y = "Relative feature contribution", fill = "") +
                scale_y_continuous(breaks = seq(-1, 1, by = 0.5), limits = c(-1, 1), expand = c(0.0, 0.0)) +
                scale_fill_manual(values = c("positive" = "#3E6F9F", "negative" = "#B0D7F0"), breaks = c("positive", "negative")) +
                geom_hline(yintercept = 0, color = "black") +
                theme(axis.text = element_text(color = "black"),
                        axis.ticks = element_line(color = "black"))
                dpi_web <- 300
                width_pixels <- 1500
                height_pixels <- 1500
                width_inches_web <- width_pixels / dpi_web
                height_inches_web <- height_pixels / dpi_web
                ggsave(filename="{image_name}", plot=pl, width=width_inches_web, height=height_inches_web, dpi=dpi_web)
                ''')
    with open(os.devnull, 'w') as nullfile:
        with contextlib.redirect_stdout(nullfile), contextlib.redirect_stderr(nullfile):
            robjects.r(r_script)
    sensitivity = robjects.r['new_pred'][0]
    return round(sensitivity, 1)

    
def R_to_Py_plot_locate(people, duration, resistance, temperature, image_name):
    r_script = (f'''
                data <- read_excel("/home/ruben/Downloads/moral sensitivity survey data 4.xlsx")
                data$situation <- as.factor(data$situation)
                data$temperature <- as.factor(data$temperature)
                data_subset <- subset(data, data$situation=="2"|data$situation=="4")
                data_subset$people[data_subset$people == "0"] <- "none"
                data_subset$people[data_subset$people == "1"] <- "one"
                data_subset$people[data_subset$people == "10" |data_subset$people == "11" |data_subset$people == "2" |data_subset$people == "3" |data_subset$people == "4" |data_subset$people == "40" |data_subset$people == "5"] <- "multiple"
                data_subset <- data_subset[data_subset$people != "clear",]
                data_subset$people <- factor(data_subset$people, levels = c("none","unclear","one","multiple"))
                data_subset <- data_subset %>% drop_na(duration)
                fit <- lm(sensitivity ~ people + duration + resistance + temperature, data = data_subset[-c(220,195,158,126,121,76),])
                pred_data <- subset(data_subset[-c(220,195,158,126,121,76),], select = c("people", "duration", "resistance", "temperature", "sensitivity"))
                explainer <- shapr(pred_data, fit)
                p <- mean(pred_data$sensitivity)
                new_data <- data.frame(resistance = c({resistance}),
                                        temperature = c("{temperature}"),
                                        people = c("{people}"),
                                        duration = c({duration}))
                new_data$temperature <- factor(new_data$temperature, levels = c("close", "higher", "lower"))
                new_data$people <- factor(new_data$people, levels = c("none", "unclear", "one", "multiple"))
                new_pred <- predict(fit, new_data)
                explanation_cat <- shapr::explain(new_data, approach = "ctree", explainer = explainer, prediction_zero = p)
                shapley_values <- explanation_cat[["dt"]][,2:5]
                standardized_values <- shapley_values / sum(abs(shapley_values))
                explanation_cat[["dt"]][,2:5] <- standardized_values
                pl <- plot(explanation_cat, digits = 1, plot_phi0 = FALSE) 
                pl[["data"]]$header <- paste("predicted sensitivity = ", round(new_pred, 1), sep = " ")
                data_plot <- pl[["data"]]
                min <- 'min.'
                temp <- NA
                if ("{temperature}" == 'close') {{
                    temp <- '<≈ thresh.'
                }}
                if ("{temperature}" == 'lower') {{
                    temp <- '&lt; thresh.'
                }}
                if ("{temperature}" == 'higher') {{
                    temp <- '&gt; thresh.'
                }}
                labels <- c(duration = paste("<img src='/home/ruben/xai4mhc/Icons/duration_fire_black.png' width='32' /><br>\n", new_data$duration, min), 
                resistance = paste("<img src='/home/ruben/xai4mhc/Icons/fire_resistance_black.png' width='38' /><br>\n", new_data$resistance, min), 
                temperature = paste("<img src='/home/ruben/xai4mhc/Icons/celsius_transparent.png' width='43' /><br>\n", temp), 
                people = paste("<img src='/home/ruben/xai4mhc/Icons/victims.png' width='19' /><br>\n", new_data$people))
                data_plot$variable <- reorder(data_plot$variable, -abs(data_plot$phi))
                pl <- ggplot(data_plot, aes(x = variable, y = phi, fill = ifelse(phi >= 0, "positive", "negative"))) +
                geom_bar(stat = "identity") +
                scale_x_discrete(name = NULL, labels = labels) +
                theme(axis.text.x = ggtext::element_markdown()) + # Removed color and size attributes here
                theme(text = element_text(size = 12, family = "sans-serif"), # Changed size to 12, which is roughly 1rem
                        plot.title = element_text(hjust = 0.5, size = 12, color = "black", face = "bold", margin = margin(b = 5)),
                        plot.caption = element_text(size = 12, margin = margin(t = 25), color = "black"),
                        panel.background = element_blank(),
                        axis.text = element_text(size = 12, color = "black"),
                        axis.text.y = element_text(color = "black", margin = margin(t = 5)),
                        axis.line = element_line(color = "black"),
                        axis.title = element_text(size = 12),
                        axis.title.y = element_text(color = "black", margin = margin(r = 10), hjust = 0.5),
                        axis.title.x = element_text(color = "black", margin = margin(t = 5), hjust = 0.5),
                        panel.grid.major = element_line(color = "#DAE1E7"),
                        panel.grid.major.x = element_blank()) +
                theme(legend.background = element_rect(fill = "white", color = "white"),
                        legend.key = element_rect(fill = "white", color = "white"),
                        legend.text = element_text(size = 12),
                        legend.position = "none",
                        legend.title = element_text(size = 12, face = "plain")) +
                labs(y = "Relative feature contribution", fill = "") +
                scale_y_continuous(breaks = seq(-1, 1, by = 0.5), limits = c(-1, 1), expand = c(0.0, 0.0)) +
                scale_fill_manual(values = c("positive" = "#3E6F9F", "negative" = "#B0D7F0"), breaks = c("positive", "negative")) +
                geom_hline(yintercept = 0, color = "black") +
                theme(axis.text = element_text(color = "black"),
                        axis.ticks = element_line(color = "black"))
                dpi_web <- 300
                width_pixels <- 1500
                height_pixels <- 1500
                width_inches_web <- width_pixels / dpi_web
                height_inches_web <- height_pixels / dpi_web
                ggsave(filename="{image_name}", plot=pl, width=width_inches_web, height=height_inches_web, dpi=dpi_web)
                ''')
    with open(os.devnull, 'w') as nullfile:
        with contextlib.redirect_stdout(nullfile), contextlib.redirect_stderr(nullfile):
            robjects.r(r_script)
    sensitivity = robjects.r['new_pred'][0]
    return round(sensitivity, 1)

def R_to_Py_plot_rescue(duration, resistance, temperature, distance, image_name):
    r_script = (f'''
                data <- read_excel("/home/ruben/Downloads/moral sensitivity survey data 4.xlsx")
                data$situation <- as.factor(data$situation)
                data$temperature <- as.factor(data$temperature)
                data$distance <- as.factor(data$distance)
                data_subset <- subset(data, data$situation=="1"|data$situation=="8")
                data_subset$people <- as.numeric(data_subset$people)
                data_subset <- subset(data_subset, (!data_subset$temperature=="close"))
                data_subset <- data_subset %>% drop_na(distance)
                fit <- lm(sensitivity ~ duration + resistance + temperature + distance, data = data_subset[-c(237,235,202,193,114,108,58,51,34,28,22),])
                pred_data <- subset(data_subset[-c(237,235,202,193,114,108,58,51,34,28,22),], select = c("duration", "resistance", "temperature", "distance", "sensitivity"))
                pred_data$temperature <- factor(pred_data$temperature, levels = c("higher", "lower"))
                explainer <- shapr(pred_data, fit)
                p <- mean(pred_data$sensitivity)
                new_data <- data.frame(duration = c({duration}), 
                                        resistance = c({resistance}),
                                        temperature = c("{temperature}"),
                                        distance = c("{distance}"))

                new_data$temperature <- factor(new_data$temperature, levels = c("higher", "lower"))
                new_data$distance <- factor(new_data$distance, levels = c("large", "small"))
                new_pred <- predict(fit, new_data)
                explanation_cat <- shapr::explain(new_data, approach = "ctree", explainer = explainer, prediction_zero = p)
                shapley_values <- explanation_cat[["dt"]][,2:5]
                standardized_values <- shapley_values / sum(abs(shapley_values))
                explanation_cat[["dt"]][,2:5] <- standardized_values
                pl <- plot(explanation_cat, digits = 1, plot_phi0 = FALSE) 
                pl[["data"]]$header <- paste("predicted sensitivity = ", round(new_pred, 1), sep = " ")
                levels(pl[["data"]]$sign) <- c("positive", "negative")
                data_plot <- pl[["data"]]
                min <- 'min.'
                temp <- NA
                if ("{temperature}" == 'close') {{
                    temp <- '<≈ thresh.'
                }}
                if ("{temperature}" == 'lower') {{
                    temp <- '&lt; thresh.'
                }}
                if ("{temperature}" == 'higher') {{
                    temp <- '&gt; thresh.'
                }}
                labels <- c(duration = paste("<img src='/home/ruben/xai4mhc/Icons/duration_fire_black.png' width='31' /><br>\n", new_data$duration, min), 
                resistance = paste("<img src='/home/ruben/xai4mhc/Icons/fire_resistance_black.png' width='38' /><br>\n", new_data$resistance, min), 
                temperature = paste("<img src='/home/ruben/xai4mhc/Icons/celsius_transparent.png' width='43' /><br>\n", temp), 
                distance = paste("<img src='/home/ruben/xai4mhc/Icons/distance_fire_victim_black.png' width='54' /><br>\n", new_data$distance))
                data_plot$variable <- reorder(data_plot$variable, -abs(data_plot$phi))
                pl <- ggplot(data_plot, aes(x = variable, y = phi, fill = ifelse(phi >= 0, "positive", "negative"))) +
                geom_bar(stat = "identity") +
                scale_x_discrete(name = NULL, labels = labels) +
                theme(axis.text.x = ggtext::element_markdown()) + # Removed color and size attributes here
                theme(text = element_text(size = 12, family = "sans-serif"), # Changed size to 12, which is roughly 1rem
                        plot.title = element_text(hjust = 0.5, size = 12, color = "black", face = "bold", margin = margin(b = 5)),
                        plot.caption = element_text(size = 12, margin = margin(t = 25), color = "black"),
                        panel.background = element_blank(),
                        axis.text = element_text(size = 12, color = "black"),
                        axis.text.y = element_text(color = "black", margin = margin(t = 5)),
                        axis.line = element_line(color = "black"),
                        axis.title = element_text(size = 12),
                        axis.title.y = element_text(color = "black", margin = margin(r = 10), hjust = 0.5),
                        axis.title.x = element_text(color = "black", margin = margin(t = 5), hjust = 0.5),
                        panel.grid.major = element_line(color = "#DAE1E7"),
                        panel.grid.major.x = element_blank()) +
                theme(legend.background = element_rect(fill = "white", color = "white"),
                        legend.key = element_rect(fill = "white", color = "white"),
                        legend.text = element_text(size = 12),
                        legend.position = "none",
                        legend.title = element_text(size = 12, face = "plain")) +
                labs(y = "Relative feature contribution", fill = "") +
                scale_y_continuous(breaks = seq(-1, 1, by = 0.5), limits = c(-1, 1), expand = c(0.0, 0.0)) +
                scale_fill_manual(values = c("positive" = "#3E6F9F", "negative" = "#B0D7F0"), breaks = c("positive", "negative")) +
                geom_hline(yintercept = 0, color = "black") +
                theme(axis.text = element_text(color = "black"),
                        axis.ticks = element_line(color = "black"))
                dpi_web <- 300
                width_pixels <- 1500
                height_pixels <- 1500
                width_inches_web <- width_pixels / dpi_web
                height_inches_web <- height_pixels / dpi_web
                ggsave(filename="{image_name}", plot=pl, width=width_inches_web, height=height_inches_web, dpi=dpi_web)
                ''')
    with open(os.devnull, 'w') as nullfile:
        with contextlib.redirect_stdout(nullfile), contextlib.redirect_stderr(nullfile):
            robjects.r(r_script)
    sensitivity = robjects.r['new_pred'][0]
    return round(sensitivity, 1)
    
# move to utils file and call once when running main.py
def load_R_to_Py():
    r_script = (f'''
                # Load libraries
                library('readxl')
                library('ggplot2')
                library('dplyr')
                library('rstatix')
                library('ggpubr')
                library('tidyverse')
                library("gvlma")
                library('lme4')
                library('shapr')
                library('ggtext')
                library('ggdist')
                ''')
    robjects.r(r_script)
    
def add_object(locs, image, size, opacity, name):
    action_kwargs = {}
    add_objects = []
    for loc in locs:
        obj_kwargs = {}
        obj_kwargs['location'] = loc
        obj_kwargs['img_name'] = image
        obj_kwargs['visualize_size'] = size
        obj_kwargs['visualize_opacity'] = opacity
        obj_kwargs['name'] = name
        add_objects+=[obj_kwargs]
    action_kwargs['add_objects'] = add_objects
    return action_kwargs

def calculate_distances(p1, p2):
    # Unpack the coordinates
    x1, y1 = p1
    x2, y2 = p2
    
    # Euclidean distance
    euclidean_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Manhattan distance
    manhattan_distance = abs(x2 - x1) + abs(y2 - y1)
    
    return euclidean_distance