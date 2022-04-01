import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.filters.Filter;

public class InfoGain {
    public static void main(String[] args) throws Exception {
        Instances instances = Utils.getInstances("./dataset/dataset.arff");

        AttributeSelection attributeSelection = new AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker ranker = new Ranker();

        //ranker.setNumToSelect(4);
        attributeSelection.setInputFormat(instances);
        attributeSelection.setEvaluator(eval);
        attributeSelection.setSearch(ranker);

        Instances instancesEval = Filter.useFilter(instances, attributeSelection);
        instancesEval.enumerateAttributes().asIterator().forEachRemaining(item->System.out.println(item));
    }
}
