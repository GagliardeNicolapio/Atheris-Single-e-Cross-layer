package crossLayer;

import atherisUtils.Utils;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.filters.Filter;

public class InfoGain {
    public static void main(String[] args) throws Exception {
        //crossLayer.InfoGain for cross-layer
        final int NUM_TO_SELECT = 7;

        Instances instances = Utils.getInstances("./dataset/datasetDataCleaningScalingFinal.arff");

        AttributeSelection attributeSelection = new AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker ranker = new Ranker();

        ranker.setNumToSelect(NUM_TO_SELECT);
        attributeSelection.setInputFormat(instances);
        attributeSelection.setEvaluator(eval);
        attributeSelection.setSearch(ranker);

        Instances instancesEval = Filter.useFilter(instances, attributeSelection);
        Utils.printCSV("./dataset/infoGainDataset.csv",instancesEval);

        instancesEval.enumerateAttributes().asIterator().forEachRemaining(item->System.out.println(item));
    }
}
