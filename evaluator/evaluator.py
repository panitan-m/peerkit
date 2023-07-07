import logging
import json
from .metrics import METRICS

logger = logging.getLogger(__name__)

class Evaluator(object):
    def __init__(self, out_dir, aspects, eval_metrics):
        self.out_dir = out_dir
        self.aspects = aspects
        self.eval_metrics = eval_metrics
        
    def save_predictions(self, preds, truths):
        with open(self.out_dir+'/predictions.json', 'w') as f:
            json.dump({
                'targets': truths.tolist(),
                'predictions': preds.tolist()
            }, f)
            
    def evaluate(self, test_pred, test_truth):
        description = '\t%20s'%('ASPECTS')
        for m in self.eval_metrics:
            description += '\t%8s'%(m.upper())
        logger.info(description)
        if len(self.aspects) == len(test_truth):
            for aid in range(len(test_truth)):
                eval_a = {}
                description = '\t%20s'%(self.aspects[aid].upper())
                for m in self.eval_metrics:
                    eval_a[m] = METRICS[m](test_pred[aid], test_truth[aid])
                    description += '\t%8.4f'%(eval_a[m])
                logger.info(description)
        else:
            eval_a = {}
            description = '\t%20s'%(self.aspects[0].upper())
            for m in self.eval_metrics:
                eval_a[m] = METRICS[m](test_pred, test_truth)
                description += '\t%8.4f'%(eval_a[m])
            logger.info(description)
        return eval_a
            