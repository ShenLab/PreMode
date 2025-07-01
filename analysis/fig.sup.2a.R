source('../utils.R')
configs <- yaml::read_yaml('../../PreMode/scripts/CHPs.v4.retrain/pretrain.seed.0.yaml')
to.plot <- get.auc.by.step(configs, base.line=F)
num_saved_batches <- 35
p <- ggplot(to.plot, aes(x=step)) +
  geom_line(aes(y=loss, col=metric_name)) +
  scale_x_continuous(breaks =
                       seq(1*configs$num_save_batches,
                           (num_saved_batches - 1)*configs$num_save_batches,
                           by = configs$num_save_batches),
                     limits = c(0, (num_saved_batches - 1)*configs$num_save_batches)) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + 
  xlab("Training Step") +
  ggtitle('Sanity Check: Training and Validation Loss') + ggeasy::easy_center_title()
ggsave('figs/fig.sup.2a.pdf', p, height = 3, width = 7.5)
