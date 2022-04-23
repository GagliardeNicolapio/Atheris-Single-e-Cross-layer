package singleLayer;

import atherisUtils.Utils;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;


public class InfoGain {
    public static void main(String[] args) throws Exception {
        final String PATH = "./dataset/datasetDataCleaningScalingFinal.arff";
        networkInfoGain(Utils.networkFeatures(PATH), 5);
        applicationInfoGain(Utils.applicationFeatures(PATH), 7);
    }

    private static void networkInfoGain(Instances instances, int numToSelect) throws Exception {
        AttributeSelection attributeSelection = new AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker ranker = new Ranker();

        ranker.setNumToSelect(numToSelect);
        attributeSelection.setInputFormat(instances);
        attributeSelection.setEvaluator(eval);
        attributeSelection.setSearch(ranker);

        Instances instancesEval = Filter.useFilter(instances, attributeSelection);
        Utils.printCSV("./dataset/infoGainDatasetSingleLayerNetwork.csv",instancesEval);
        instancesEval.enumerateAttributes().asIterator().forEachRemaining(item->System.out.println(item));
    }

    private static void applicationInfoGain(Instances instances, int numToSelect) throws Exception {
        AttributeSelection attributeSelection = new AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker ranker = new Ranker();

        ranker.setNumToSelect(numToSelect);
        attributeSelection.setInputFormat(instances);
        attributeSelection.setEvaluator(eval);
        attributeSelection.setSearch(ranker);

        Instances instancesEval = Filter.useFilter(instances, attributeSelection);
        Utils.printCSV("./dataset/infoGainDatasetSingleLayerApplication.csv",instancesEval);
        instancesEval.enumerateAttributes().asIterator().forEachRemaining(item->System.out.println(item));
    }


}
