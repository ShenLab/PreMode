library(ggplot2)
task.dic <- rbind(read.csv('single.assays.txt', header = F), read.csv('multiple.assays.txt', header = F))
# add protein gym
alphabet_premode <- c('L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
                      'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C')
genes <- task.dic$V1
# add baseline AUC
# esm alphabets
source('./AUROC.R')
alphabet <- c('<cls>', '<pad>', '<eos>', '<unk>',
              'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
              'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
              'X', 'B', 'U', 'Z', 'O', '.', '-',
              '<null_1>', '<mask>')
result <- data.frame()
protein.gym.res <- read.csv('ProteinGym/DMS_supervised_substitutions_scores.csv')
# protein.gym.res <- read.csv('/share/vault/Users/gz2294/Data/DMS/ProteinGym/benchmarks/DMS_supervised/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level_fold_random_5.csv')
for (i in 1:length(genes)) {
  for (fold in 0:4) {
    # REVEL, PrimateAI, ESM AUC
    if (file.exists(paste0('PreMode/', genes[i], '/',
                           '/testing.fold.', fold, '.csv'))) {
      test.result <- read.csv(paste0('PreMode/', genes[i], '/',
                                     '/testing.fold.', fold, '.csv'), row.names = 1)
      task.length <- 1
      train.yaml <- yaml::read_yaml(paste0('../scripts/PreMode/ProteinGym.random.5/', 
                                           genes[i], 
                                           '.5fold/', genes[i], '.fold.4.yaml'))
      train.data <- read.csv(train.yaml$data_file_train, row.names = 1)
      train.steps <- ceiling(dim(train.data)[1]/train.yaml$batch_size) * train.yaml$num_epochs
      # add Hsu et al. results
      PreMode.auc <- plot.R2(test.result$score, test.result$logits, bin = F)
    } else {
      # add hsu et al results
      PreMode.auc <- list(R2=NA)
    }
    
    protein.gym.auc <- protein.gym.res[protein.gym.res$assay_id == genes[i] & protein.gym.res$fold_variable_name == "fold_random_5",]
    # protein.gym.auc <- protein.gym.res[protein.gym.res$DMS_id == genes[i],]
    
    to.append <- data.frame(min.val.R = c(protein.gym.auc$Spearman_fitness,
                                          PreMode.auc$R2),
                            task.name = genes[i],
                            HGNC=genes[i],
                            fold=fold,
                            npoints=dim(test.result)[1],
                            ntrain=dim(train.data)[1],
                            train.steps=train.steps)
    # to.append <- data.frame(min.val.R = c(as.numeric(protein.gym.auc[2:dim(protein.gym.auc)[2]]),
    #                                       PreMode.auc$R2),
    #                         task.name = genes[i],
    #                         HGNC=genes[i],
    #                         fold=fold,
    #                         npoints=dim(test.result)[1],
    #                         ntrain=dim(train.data)[1],
    #                         train.steps=train.steps)
    to.append$model <- rep(c(protein.gym.auc$model_name, "PreMode"), each = task.length)
    # to.append$model <- rep(c(colnames(protein.gym.res)[2:dim(protein.gym.auc)[2]], "PreMode"), each = task.length)
    result <- rbind(result, to.append)
  }
}
num.models <- length(unique(result$model))
# filter out tasks with incomplete models
na.tasks <- c()
for (task in unique(result$task.name)) {
  result.task <- result[result$task.name == task,]
  # number of other models
  if (length(unique(result.task$model[!is.na(result.task$min.val.R) & result.task$model != 'PreMode'])) != 10) {
  # if (length(unique(result.task$model[!is.na(result.task$min.val.R) & result.task$model != 'PreMode'])) != 11) {
    na.tasks <- c(na.tasks, task)
  }
}
# na.tasks <- c(na.tasks,
#               'B2L11_HUMAN_Dutta_2010_binding-Mcl-1',
#               'CASP3_HUMAN_Roychowdhury_2020',
#               'CASP7_HUMAN_Roychowdhury_2020',
#               'NPC1_HUMAN_Erwood_2022_RPE1')
result.noNA <- result[!result$task.name %in% na.tasks,]
# result.noNA <- result
# plot the task weighted averages as well as task size weighted error bars
uniq.result.plot <- result.noNA[result.noNA$fold==0,]
for (i in 1:dim(uniq.result.plot)[1]) {
  uniq.result.plot$rho[i] = mean(result.noNA$min.val.R[result.noNA$model==uniq.result.plot$model[i] & 
                                                         result.noNA$task.name==uniq.result.plot$task.name[i]], na.rm=T)
  uniq.result.plot$rho.sd[i] = sd(result.noNA$min.val.R[result.noNA$model==uniq.result.plot$model[i] & 
                                                          result.noNA$task.name==uniq.result.plot$task.name[i]], na.rm=T)
}
# aggregate across models
uniq.model.result.plot <- uniq.result.plot[!duplicated(uniq.result.plot[,c('model')]),]
for (i in 1:dim(uniq.model.result.plot)[1]) {
  # do weighted average based on number of points
  rhos <- uniq.result.plot$rho[uniq.result.plot$model == uniq.model.result.plot$model[i]]
  rho.sds <- uniq.result.plot$rho.sd[uniq.result.plot$model == uniq.model.result.plot$model[i]]
  npoints <- uniq.result.plot$npoints[uniq.result.plot$model == uniq.model.result.plot$model[i]]
  uniq.model.result.plot$stab.rho[i] <- sum(rhos*npoints)/sum(npoints)
  uniq.model.result.plot$stab.rho.sd[i] <- sum(rho.sds*npoints)/sum(npoints)
  # uniq.model.result.plot$stab.rho[i] <- mean(rhos)
  # uniq.model.result.plot$stab.rho.sd[i] <- mean(rho.sds)
  
}
# Extract unique model names excluding "PreMode"
other_models <- setdiff(unique(result.noNA$model), "PreMode")
other_models <- sort(other_models)
# Generate automatic colors for other models
auto_colors <- scales::hue_pal()(length(other_models))
# Combine the colors for "PreMode" and the rest
color_mapping <- c("PreMode" = "black", setNames(auto_colors, other_models))
p <- ggplot(result.noNA, aes(y=min.val.R, x=task.name, col=model)) +
  geom_point(alpha=0) +
  # stat_summary(data = result.noNA,
  #              aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
  #                  y = min.val.R, col=model), 
  #              fun.data = mean_se, geom = "errorbar", width = 0.2) +
  stat_summary(data = result.noNA, 
               aes(x=as.numeric(factor(task.name))+0.4*(as.numeric(factor(model)))/num.models-0.2*(num.models+1)/num.models,
                   y = min.val.R, col=model), 
               fun.data = mean_se, geom = "point") +
  geom_hline(data = uniq.model.result.plot, aes(yintercept=stab.rho, col=model), linetype='dashed') +
  labs(x = "task", y = "min.val.R", fill = "model") +
  theme_bw() + 
  theme(axis.text.x = element_text(angle=60, vjust = 1, hjust = 1), 
        legend.position="bottom", 
        legend.direction="horizontal") +
  # ylim(-1, 1) +
  # coord_flip() + 
  guides(col=guide_legend(ncol=1)) + ggtitle('Transfer Learning Compare on ProteinGym') +
  ggeasy::easy_center_title() +
  xlab('task: ProteinGym Human Proteins DMS') + ylab('Spearman Rho') +
  scale_color_manual(values = color_mapping) + 
  coord_flip() + guides(col=guide_legend(nrow=4),
                        shape=guide_legend(nrow=4))
ggsave(paste0('figs/fig.proteingym.a.pdf'), p, height = 15, width = 12)
saveRDS(result, 'figs/fig.proteingym.RDS')
