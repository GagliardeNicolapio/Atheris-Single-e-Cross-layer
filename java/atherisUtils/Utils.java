package atherisUtils;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;

public class Utils {

    //ritorna le istanze lette dal path
    public static Instances getInstances(String path) throws Exception {
        ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource(path);
        Instances instances = dataSource.getDataSet();
        //System.out.println(instances.attribute(instances.numAttributes()-1));
        instances.setClassIndex(instances.numAttributes()-1);
        return instances;
    }

    //stampa in csv le istanze passate
    public static void printCSV(String nameFile, Instances instances) throws IOException {
        FileWriter fileWriter = new FileWriter(nameFile);
        fileWriter.write(Utils.getAttributesCSV(instances)+"\n");
        for(int i=0; i<instances.numInstances(); i++)
            fileWriter.write(instances.get(i).toString()+"\n");

        fileWriter.flush();
        fileWriter.close();
    }

    //ritorna la stringa "colonna1,colonna2,colonna3..."
    public static String getAttributesCSV(Instances instances){
        String columns="";
        Iterator<Attribute> iterator = instances.enumerateAttributes().asIterator();
        while(iterator.hasNext())
            columns+=iterator.next().name()+",";
        columns+="Type";
        System.out.println(columns);
        return columns;
    }

    public static Instances networkFeatures(String path) throws Exception {
        Instances instancesNetwork = Utils.getInstances(path);
        Remove removeFilter = new Remove();
        int[] networkFeatures = {9,10,11,12,13,14,15,16,17,19};
        removeFilter.setAttributeIndicesArray(networkFeatures);
        removeFilter.setInvertSelection(true);
        removeFilter.setInputFormat(instancesNetwork);
        return Filter.useFilter(instancesNetwork, removeFilter);
    }

    public static Instances applicationFeatures(String path) throws Exception {
        Instances instancesApplication = Utils.getInstances(path);
        Remove removeFilter = new Remove();
        int[] applicationFeatures = {0,1,2,3,4,5,6,7,8,18,19};
        removeFilter.setAttributeIndicesArray(applicationFeatures);
        removeFilter.setInvertSelection(true);
        removeFilter.setInputFormat(instancesApplication);
        return Filter.useFilter(instancesApplication, removeFilter);
    }
}
