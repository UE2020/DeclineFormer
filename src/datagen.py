import re

def create_tab_separated_file(english_paragraph, latin_paragraph, filename):
    # Split the paragraphs into sentences
    english_sentences = re.split('[.!?]', english_paragraph)
    latin_sentences = re.split('[.!?]', latin_paragraph)

    # Remove any empty sentences
    english_sentences = [sentence.strip() for sentence in english_sentences if sentence.strip()]
    latin_sentences = [sentence.strip() for sentence in latin_sentences if sentence.strip()]

    # Check if both paragraphs have the same number of sentences
    # if len(english_sentences) != len(latin_sentences):
    #     raise ValueError("The number of sentences in the English and Latin paragraphs do not match.")

    # Write the sentences to a tab-separated file
    with open(filename, 'w') as f:
        for english, latin in zip(english_sentences, latin_sentences):
            f.write(f"{latin}\t{english}\n")

lt = """
quamquam mihi semper frequens conspectus vester multo iucundissimus, hic autem locus ad agendum amplissimus, ad dicendum ornatissimus est visus, Quirites, tamen hoc aditu laudis qui semper optimo cuique maxime patuit non mea me voluntas adhuc sed vitae meae rationes ab ineunte aetate susceptae prohibuerunt. nam cum antea nondum huius auctoritatem loci attingere auderem statueremque nihil huc nisi perfectum ingenio, elaboratum industria adferri oportere, omne meum tempus amicorum temporibus transmittendum putavi.
ita neque hic locus vacuus fuit umquam ab eis qui vestram causam defenderent et meus labor in privatorum periculis caste integreque versatus ex vestro iudicio fructum est amplissimum consecutus. nam cum propter dilationem comitiorum ter praetor primus centuriis cunctis renuntiatus sum, facile intellexi, Quirites, et quid de me iudicaretis et quid aliis praescriberetis. nunc cum et auctoritatis in me tantum sit quantum vos honoribus mandandis esse voluistis, et ad agendum facultatis tantum quantum homini vigilanti ex forensi usu prope cotidiana dicendi exercitatio potuit adferre, certe et, si quid auctoritatis in me est, apud eos utar qui eam mihi dederunt et, si quid in dicendo consequi possum, eis ostendam potissimum qui ei quoque rei fructum suo iudicio tribuendum esse duxerunt.

atque illud in primis mihi laetandum iure esse video quod in hac insolita mihi ex hoc loco ratione dicendi causa talis oblata est in qua oratio deesse nemini possit. dicendum est enim de Cn. Pompei singulari eximiaque virtute; huius autem orationis difficilius est exitum quam principium invenire. ita mihi non tam copia quam modus in dicendo quaerendus est.

atque ut inde oratio mea proficiscatur unde haec omnis causa ducitur, bellum grave et periculosum vestris vectigalibus atque sociis a duobus potentissimis adfertur regibus, Mithridate et Tigrane, quorum alter relictus, alter lacessitus occasionem sibi ad occupandam Asiam oblatam esse arbitratur. equitibus Romanis, honestissimis viris, adferuntur ex Asia cotidie litterae, quorum magnae res aguntur in vestris vectigalibus exercendis occupatae; qui ad me pro necessitudine quae mihi est cum illo ordine causam rei publicae periculaque rerum suarum detulerunt,
Bithyniae quae nunc vestra provincia est vicos exustos esse compluris, regnum Ariobarzanis quod finitimum est vestris vectigalibus totum esse in hostium potestate; L.(ucium) Lucullum magnis rebus gestis ab eo bello discedere; huic qui successerit, non satis esse paratum ad tantum bellum administrandum; unum ab omnibus sociis et civibus ad id bellum imperatorem deposci atque expeti, eundem hunc unum ab hostibus metui, praeterea neminem.
"""

en = """
Although, O Romans, your numerous assembly has always seemed to me the most agreeable body that any one can address, and this place, which is most honourable to plead in, has also seemed always the most distinguished place for delivering an oration in, still I have been prevented from trying this road to glory, which has at all times been entirely open to every virtuous man, not indeed by my own will, but by the system of life which I have adopted from my earliest years. For as hitherto I have not dared, on account of my youth, to intrude upon the authority of this place, and as I considered that no arguments ought to be brought to this place except such as were the fruit of great ability, and worked up with the greatest industry, I have thought it fit to devote all my time to the necessities of my friends.
And accordingly, this place has never been unoccupied by men who were defending your cause, and my industry, which has been virtuously and honestly employed about the dangers of private individuals, has received its most honourable reward in your approbation. For when, on account of the adjournment of the comitia, I was three times elected the first praetor by all the centuries, I easily perceived, O Romans, what your opinion of me was, and what conduct you enjoined to others. Now, when there is that authority in me which you, by conferring honours on me, have chosen that there should be, and all that facility in pleading which almost daily practice in speaking can give a vigilant man who has habituated himself to the forum, at all events, if I have any authority, I will employ it before those who have given it to me; and if I can accomplish anything by speaking, I will display it to those men above all others, who have thought fit, by their decision, to confer honours on that qualification.

And, above all things, I see that I have reason to rejoice on this account, that, since I am speaking in this place, to which I am so entirely unaccustomed, I have a cause to advocate in which eloquence can hardly fail any one; for I have to speak of the eminent and extraordinary virtue of Cnaeus Pompey; and it is harder for me to find out how to end a discourse on such a subject, than how to begin one. So that what I have to seek for is not so much a variety of arguments, as moderation in employing them.

And, that my oration may take its origin from the same source from which all this cause is to be maintained; an important war, and one perilous to your revenues and to your allies, is being waged against you by two most powerful kings, Mithridates and Tigranes. One of these having been left to himself, and the other having been attacked, thinks that an opportunity offers itself to him to occupy all Asia. Letters are brought from Asia every day to Roman knights, most honourable men, who have great property at stake, which is all employed in the collection of your revenues; and they, in consequence of the intimate connection which I have with their order, have come to me and entrusted me with the task of pleading the cause of the republic, and warding off danger from their private fortunes.
They say that many of the villages of Bithynia, which is at present a province belonging to you, have been burnt; that the kingdom of Ariobarzanes, which borders on those districts from which you derive a revenue, is wholly in the power of the enemy; that Lucullus, after having performed great exploits, is departing from that war; that it is not enough that whoever succeeds him should be prepared for the conduct of so important a war; that one general is demanded and required by all men, both allies and citizens, for that war; that he alone is feared by the enemy, and that no one else is.
"""
create_tab_separated_file(en, lt, "pairs.tsv")