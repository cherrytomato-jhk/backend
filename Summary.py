import tensorflow as tf
import pickle
from model import Model
import numpy as np
import re

class Summarization():
    def __init__(self):
        self._load_data()
        self._set_model()
        
    def _load_data(self):
        self.article_max_len = 350 #CHANGE
        self.summary_max_len = 55 #CHANGE

        with open("args.pickle", "rb") as f:
            self.args = pickle.load(f)

        with open("word_dict.pickle", "rb") as f:
            self.word_dict = pickle.load(f)
        self.reversed_dict = dict(zip(self.word_dict.values(), self.word_dict.keys()))

    def _set_model(self):
        self.sess = tf.Session()
        self.model = Model(self.reversed_dict, self.article_max_len, self.summary_max_len, self.args, forward_only=True)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.ckpt = tf.train.get_checkpoint_state("./saved_model/")
        self.saver = self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)


    def _batch_iter(self, inputs, outputs, batch_size, num_epochs):
        inputs = np.array(inputs)
        outputs = np.array(outputs)

        num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
        for epoch in range(num_epochs):
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, len(inputs))
                yield inputs[start_index:end_index], outputs[start_index:end_index]


    def get_result(self, article_list):
        # preprocessing input article
        article_list = re.sub(r"([?.!,¿])", r" \1 ", article_list)
        article_list = re.sub(r'[" "]+', " ", article_list)
        article_list = re.sub(r"[^a-zA-Z0-9?.!,@%$¿]+", " ", article_list)
        article_list.strip().lower()
        article_list = [article_list]

        x = []
        for article in article_list:
            x.append(article.split())
        x = [[self.word_dict.get(w, self.word_dict["<unk>"]) for w in d] for d in x]
        x = [d[:self.article_max_len] for d in x]
        x = [d + (self.article_max_len - len(d)) * [self.word_dict["<padding>"]] for d in x]
        
        batches = self._batch_iter(x, [0] * len(x), self.args.batch_size, 1)

        for batch_x, _ in batches:
            batch_x_len = [len([y for y in x if y != 0]) for x in batch_x]

            valid_feed_dict = {
                self.model.batch_size: len(batch_x),
                self.model.X: batch_x,
                self.model.X_len: batch_x_len,
            }

            prediction = self.sess.run(self.model.prediction, feed_dict=valid_feed_dict)
            prediction_output = [[self.reversed_dict[y] for y in x] for x in prediction[:, 0, :]]

            summary = []
            for word in prediction_output[0]:
                    if word == "</s>":
                        break
                    if word not in summary:
                        summary.append(word)

            output = " ".join(summary)
        return output


if __name__ == "__main__":
    summarizer = Summarization()
    article_list = 'The worst oil spill in U.S. history has heated up the debate about how to feed the nation with cleaner, safer energy.The argument over whether to "drill baby drill" — the 2008 GOP campaign mantra — returned in April after a deadly oil rig blast spilled millions of gallons of oil into the Gulf of Mexico. The disaster threatened jobs in the nation\'s seafood and tourism industries and hundreds of miles of sensitive wetlands, marshes and wildlife.The spill came as President Obama was backing expanded offshore oil drilling and as Congress considered wide-ranging energy legislation aimed at cutting pollution and U.S. dependency on foreign oil.Before BP\'s Gulf oil disaster, Obama\'s late March proposal called for opening swaths of U.S. coastal waters in the Atlantic Ocean and the Gulf of Mexico to oil and natural gas drilling. After the spill, Obama said offshore drilling is key to "our energy future," but it can move forward only with assurances that the Gulf disaster won\'t be repeated. The president halted drilling permits for new wells and ordered inspections of all deepwater operations.Former GOP vice presidential nominee and Alaska Gov. Sarah Palin — who in 2008 called on the U.S. to "drill baby drill" in the protected Arctic National Wildlife Refuge — tweeted that Obama should offset his drilling moratorium by "correspondingly" allowing "more onshore drilling, including ANWR reserves. Domestic oil\'s still required in US industry."Will the spill change Obama\'s oil policy?Senate Democrats gave up on their goal of passing a slimmed-down energy bill before August recess.Senate Majority Leader Harry Reid blamed unified Republican opposition to the proposal. "It\'s a sad day when you can\'t find a handful of Republicans to support a bill that would create 70,000 clean-energy jobs, hold BP accountable, and look at a future as it relates to what BP did."Republicans pointed at Reid."Sen. Reid is predictably blaming Republicans for standing in the way of a bill that he threw together in secret and without input from almost any other member of the Senate," said Sen. Lisa Murkowski of Alaska, the top Republican on the energy committee. "Process alone guaranteed its failure, although substance would have as well had Sen. Reid actually brought his bill up for debate or a vote."The Democratic bill had been stripped of its most controversial components such as limits on carbon emissions.The remaining legislation would have eliminated a $75-million cap on economic damages that offshore oil drillers would be responsible for. It would have reformed federal government oversight of offshore drilling. It encouraged the use of natural gas engines in commercial trucks; and it promoted high efficiency appliances in homes.Republicans planned to offer an alternative bill that included lifting the Obama administration\'s blanket moratorium on deepwater drilling, setting up a bipartisan commission with subpoena power to investigate the Gulf oil disaster and propose reforms, and allowing for revenue sharing for states that permit offshore drilling.An energy bill passed last year by the House of Representatives included a controversial cap-and-trade system. The system would set a price for greenhouse gas emissions, such as carbon dioxide, and polluters would obtain and trade credits for emissions over a set threshold. Republicans and oil and coal producers oppose such a plan. Supporters say it\'s the best way to begin reducing U.S. dependence on fossil fuels.Would a nuclear power comeback be a good thing?Obama has been a key figure in energy talks among lawmakers. In late June, he met with a bipartisan group of senators at the White House to discuss passing an energy and climate change bill this year. The president had expressed hope that something could be done. "There was agreement on the sense of urgency required to move forward with legislation, and the president is confident that we will be able to get something done this year," the White House said in a statement.Obama told senators that he believed the best way to make a transition to a clean-energy economy is with a bill that "makes clean energy the profitable kind of energy for America\'s businesses by putting a price on pollution," the statement said.CNN Congressional Producer Ted Barrett contributed to this report.'
    results = summarizer.get_result(article_list)
    print(results)