import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;

public class Utils {

    //ritorna le istanze lette dal path
    public static Instances getInstances(String path) throws Exception {
        ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource(path);
        Instances instances = dataSource.getDataSet();
        System.out.println(instances.attribute(instances.numAttributes()-1));
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
}
