import math
import torch
import gc
import sys

from torch import optim

from pythia.utils.flags import flags
from pythia.utils.configuration import Configuration
from pythia.utils.checkpoint import Checkpoint
from pythia.utils.logger import Logger
from pythia.utils.general import lr_lambda_update, clip_gradients, \
                get_optimizer_parameters
from pythia.utils.build import build_model
from pythia.utils.timer import Timer
from pythia.utils.early_stopping import EarlyStopping
from pythia.core.task_loader import TaskLoader
from pythia.modules.losses import Loss


class Trainer:
    def __init__(self):
        parser = flags.get_parser()
        self.args = parser.parse_args()

    def load(self):
        self.load_config()
        self.load_task()

        self.writer = Logger(self.config)

        self.load_model()
        self.load_optimizer()
        self.load_extras()

    def load_config(self):
        # TODO: Review configuration update once again
        # (remember clip_gradients case)
        self.configuration = Configuration(self.args.config)
        self.configuration.update_with_args(self.args, force=True)

        self.config = self.configuration.get_config()

        self.config_based_setup()

    def load_task(self):
        self.task_loader = TaskLoader(self.config)
        self.configuration.update_with_task_config(self.task_loader)

        # Update with the config override if passed
        self.configuration.override_with_cmd_config(self.args.config_override)

        # Update with args once again as they are the most important
        self.configuration.update_with_args(self.args)
        self.configuration.pretty_print()

        self.task_loader.load_dataset()
        self.task_loader.make_dataloaders()

        self.train_loader = self.task_loader.train_loader
        self.dev_loader = self.task_loader.dev_loader
        self.test_loader = self.task_loader.test_loader

        self.train_meter = self.task_loader.train_meter
        self.dev_meter = self.task_loader.dev_meter
        self.test_meter = self.task_loader.test_meter

    def load_model(self):
        attributes = self.config['model_attributes'][self.config['model']]
        attributes['model'] = self.config['model']

        data_root_dir = self.config['task_attributes']['data_root_dir']
        attributes['data_root_dir'] = data_root_dir

        self.task_loader.update_config_for_model(attributes)
        self.model = build_model(attributes)
        self.task_loader.clean_config(attributes)

        if self.config['use_cuda']:
            self.model = self.model.cuda()

        if self.config['use_cuda'] and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    def load_optimizer(self):
        optimizer_method = self.config['optimizer_attributes']['type']
        optimizer_class = getattr(optim, optimizer_method)

        parameters = get_optimizer_parameters(self.model, self.config)
        rest_optimizer_params = self.config['optimizer_attributes']['params']
        self.optimizer = optimizer_class(parameters, **rest_optimizer_params)

    def load_extras(self):
        self.criterion = Loss(self.config['loss'])

        self.checkpoint = Checkpoint(self)

        monitored_metric = self.config['task_attributes']['monitored_metric']
        metric_minimize = self.config['task_attributes']['metric_minimize']
        should_early_stop = self.config['task_attributes']['should_early_stop']
        patience = self.config['training_parameters']['patience']
        self.early_stopping = EarlyStopping(self.model,
                                            self.checkpoint,
                                            self.dev_meter,
                                            monitored_metric,
                                            patience=patience,
                                            minimize=metric_minimize,
                                            should_stop=should_early_stop)

        self.current_epoch = 0
        self.current_iteration = 0

        self.checkpoint.load_state_dict()

        self.lr_scheduler = None
        if self.config['lr_scheduler'] is True:
            scheduler_class = optim.lr_scheduler.LambdaLR
            scheduler_func = lambda x: lr_lambda_update(x, self.config)
            self.lr_scheduler = scheduler_class(self.optimizer,
                                                lr_lambda=scheduler_func)

    def config_based_setup(self):
        torch.manual_seed(self.config['seed'])

        if self.config['use_cuda']:
            torch.cuda.manual_seed(self.config['seed'])

    def train(self):
        training_parameters = self.config['training_parameters']
        log_interval = training_parameters['log_interval']
        snapshot_interval = training_parameters['snapshot_interval']
        max_iterations = training_parameters['max_iterations']
        should_clip_gradients = training_parameters['clip_gradients']
        max_epochs = self.config['max_epochs']

        self.writer.write(self.model)
        should_check_on_epoch = False

        if max_epochs is not None:
            should_check_on_epoch = True
            max_iterations = math.inf

        self.model.train()
        self.train_timer = Timer()
        self.snapshot_timer = Timer()

        while self.current_iteration < max_iterations:
            self.current_epoch += 1
            if should_check_on_epoch and self.current_epoch > max_epochs:
                break

            for batch in self.train_loader:
                self.current_iteration += 1

                if self.current_iteration > max_iterations:
                    break

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(self.current_iteration)

                self.optimizer.zero_grad()

                data, y = self.task_loader.prepare_batch('train', batch)

                # Arguments should be a dict at this point
                output = self.model(**data)
                loss = self.criterion(output, y)

                loss.backward()

                if should_clip_gradients:
                    clip_gradients(self.model, self.current_iteration,
                                   self.writer, self.config)

                self.optimizer.step()
                self.train_meter(output, y)

                extra_info = None
                should_print = self.current_iteration % log_interval == 0

                # TODO: Move in separate function
                if should_print is True:
                    sys.stdout.flush()
                    extra_info = self.single_batch_eval(self.dev_loader,
                                                        self.dev_meter)
                    time_taken = self.train_timer.get_time_since_start()
                    extra_info += ", time: %s" % time_taken

                # Don't print train metrics if it is not log interval
                # so as to escape clutter
                self.task_loader.report_metrics(self.writer,
                                                self.train_meter,
                                                loss.data.item(),
                                                self.current_iteration,
                                                extra_info=extra_info,
                                                should_print=should_print)

                if should_print is True:
                    self.train_timer.reset()

                if self.current_iteration % snapshot_interval == 0:
                    # TODO: Implement early stopping
                    # TODO: Do validation check here
                    avg_loss = self.evaluate(self.dev_loader,
                                             self.dev_meter)
                    time_taken = self.snapshot_timer.get_time_since_start()
                    extra_info = ", time: %s" % time_taken
                    self.snapshot_timer.reset()
                    stop = self.early_stopping(self.current_iteration,
                                               avg_loss)
                    extra_info += "\n %s" % self.early_stopping.get_info()

                    self.task_loader.report_metrics(self.writer,
                                                    self.dev_meter,
                                                    avg_loss,
                                                    self.current_iteration,
                                                    extra_info=extra_info)

                    if stop is True:
                        self.writer.write("Early stopping activated")
                        break

                gc.collect()

        avg_test_loss = self.evaluate(self.test_loader, self.test_meter)
        self.task_loader.report_metrics(self.writer, self.test_meter,
                                        avg_test_loss, self.current_iteration)

    def single_batch_eval(self, loader, meter):
        self.model.eval()
        batch = next(iter(loader))
        meter.reset()
        data, y = self.task_loader.prepare_batch(loader.dataset_type, batch)
        output = self.model(**data)
        meter(output, y)
        loss = self.criterion(output, y)
        self.model.train()

        # TODO: Do replace in log string function itself
        return meter.get_log_string(loss).replace("Average", "Single")

    def evaluate(self, loader, meter):
        self.model.eval()

        meter.reset()
        total_loss = 0
        total_samples = 0

        for batch in loader:
            data, y = self.task_loader.prepare_batch(loader.dataset_type,
                                                     batch)
            total_samples += y.size(0)

            output = self.model(**data)

            meter(output, y)
            loss = self.criterion(output, y)
            if loss is not None:
                total_loss += loss.data.item() * y.size(0)

        self.model.train()
        return total_loss / total_samples