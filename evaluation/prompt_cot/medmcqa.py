cot_prompts = [
    {
        "prompt": """An expected side effect of creatine supplementation is:\n\nOptions:\nA. muscle weakness.\nB. gain in body mass.\nC. muscle cramps.\nD. loss of electrolytes.""",
        "gold": "B",
        "steps": [
            "We refer to Wikipedia articles on medicine for help. Creatine supplementation is a dietary supplement that results in body mass gain."
        ]
    },
    {
        "prompt": """Which of the following is not a true statement?\n\nOptions:\nA. Muscle glycogen is broken down enzymatically to glucose-1-phosphate \nB. Elite endurance runners have a high proportion of Type I fibres in their leg muscles\nC. Liver glycogen is important in the maintenance of the blood glucose concentration\nD. Insulin promotes glucose uptake by all tissues in the body""",
        "gold": "D",
        "steps": [
            "A. “Muscle glycogen is broken down enzymatically to glucose-1-phosphate”: This is a correct statement.",
            "B. “Elite endurance runners have a high proportion of Type I fibres in their leg muscles”: This is a correct statement.",
            "C. “Liver glycogen is important in the maintenance of the blood glucose concentration”: This is a correct statement. ",
            "D. “Insulin promotes glucose uptake by all tissues in the body”: This is not a correct statement, because insulin promotes glucose uptake by the liver, adipose tissue, and muscle, but not all tissues. For instance, the tissues in the brain and red blood cells are not affected by insulin."
        ]
    },
    {
        "prompt": "A high school science teacher fills a 1 liter bottle with pure nitrogen and seals the lid. The pressure is 1.70 atm, and the room temperature is 25°C. Which two variables will both increase the pressure of the system, if all other variables are held constant?\n\nOptions:\nA. Increasing temperature, increasing moles of gas \nB. Increasing temperature, increasing volume\nC. Decreasing volume, decreasing temperature\nD. Decreasing moles of gas, increasing volume",
        "gold": "A",
        "steps": [
            "We refer to Wikipedia articles on medicine for help. The relevant equation for this is the ideal gas law: PV=nRT.",
            "To increase the pressure of the system (P), either n (number of moles of the gas) or T (temperature) have to increase.",
            "The correct answer that matches these criteria is: Increasing temperature, increasing moles of gas."
        ]
    },
    {
        "prompt": """In a genetic test of a newborn, a rare genetic disorder is found that has X-linked recessive transmission. Which of the following statement is likely true regarding the pedigree of this disorder?\n\nOptions:\nA. All descendants on the maternal side will have the disorder. \nB. Females will be approximately twice as affected as males in this family.\nC. All daughters of an affected male will be affected.\nD. There will be equal distribution of males and females affected.""",
        "gold": "C",
        "steps": [
            "Let's recall first that females have two X chromosomes, while males have one X and one Y chromosome. This is an important fact we need to know before answering this question.",
            "Because a male can only pass his only one X chromosome to a daughter, if he is affected by this rare genetic disorder, then we know for sure that he will pass this rare genetic disorder to all his future-born daughters.",
            "Therefore, \nC.: All daughters of an affected male will be affected” is a correct statement."
        ]   
    },
    {
        "prompt": """Glucose is transported into the muscle cell:\n\nOptions:\nA. via protein transporters called GLUT4. \nB. only in the presence of insulin.\nC. via hexokinase.\nD. via monocarbylic acid transporters.""",
        "gold": "A",
        "steps": [
            "Glucose (also known as the blood sugar) is the main sugar found in the human body.",
            "It is transported into the muscle cell via diffusion through protein transporters called GLUT4.",
        ]
    },
    {
        "prompt": """Glycolysis is the name given to the pathway involving the conversion of:\n\nOptions:\nA. glycogen to glucose-1-phosphate. \nB. glycogen or glucose to fructose. \nC. glycogen or glucose to pyruvate or lactate. \nD. glycogen or glucose to pyruvate or acetyl CoA.""",
        "gold": "C",
        "steps": [
            "Glycolysis is the name given to the pathway involving conversion of glycogen or glucose to pyruvate or lactate."
        ]
    },
    {
        "prompt": """What is the difference between a male and a female catheter?\n\nOptions:\nA. Male and female catheters are different colours. \nB. Male catheters are longer than female catheters. \nC. Male catheters are bigger than female catheters. \nD. Female catheters are longer than male catheters.""",
        "gold": "B",
        "steps": [
               "The difference between a male and female catheter is that male catheters tend to be longer than female catheters."
        ]
    },
    {
        "prompt": """How many attempts should you make to cannulate a patient before passing the job on to a senior colleague, according to the medical knowledge of 2020?\n\nOptions:\nA. 4 \nB. 3 \nC. 2 \nD. 1""",
        "gold": "C",
        "steps": [
            "According to the medical protocol as of 2020, you should make two attempts to cannulate a patient before passing the job on to a more-senior practitioner."
        ]
    },
    {
        "prompt": """In the assessment of the hand function which of the following is true?\n\nOptions:\nA. Abduction of the thumb is supplied by spinal root T2 \nB. Opposition of the thumb by opponens policis is supplied by spinal root T1 \nC. Finger adduction is supplied by the median nerve \nD. Finger abduction is mediated by the palmar interossei""",
        "gold": "B",
        "steps": [
            "Of all the options, it is only true that the opposition of the thumb by opponens pollicis is supplied by spinal root T1."
        ]
    },
    {
        "prompt": """The energy for all forms of muscle contraction is provided by:\n\nOptions:\nA. ATP \nB. ADP \nC. phosphocreatine \nD. oxidative phosphorylation""",
        "gold": "A",
        "steps": [
            "The energy for muscular contraction is provided by ATP (adenosine triphosphate), which is the powerhouse of the cell."
        ]
    },
    {
        "prompt": """Which of the following represents an accurate statement concerning arthropods?\n\nOptions:\nA. They possess an exoskeleton composed primarily of peptidoglycan. \nB. They possess an open circulatory system with a dorsal heart. \nC. They are members of a biologically unsuccessful phylum incapable of exploiting diverse habitats and nutrition sources. \nD. They lack paired, jointed appendages.""",
        "gold": "B",
        "steps": [
            "Peptidoglycan is known to comprise the plasma membrane of most bacteria, rather than the exoskeleton of arthropods, which is made of chitin, which rules out (A).",
            "The answer (C) is false because arthropods are a highly successful phylum.",
            "Likewise, arthropods have paired, jointed appendages, which rules out (D).",
            "The only remaining option is (B), as arthropods have an open circulatory system with a dorsal tubular heart."
        ]
    },
    {
        "prompt": """In a given population, 1 out of every 400 people has a cancer caused by a completely recessive allele, b. Assuming the population is in Hardy-Weinberg equilibrium, which of the following is the expected proportion of individuals who carry the b allele but are not expected to develop the cancer?\n\nOptions:\nA. 1/400 \nB. 19/400 \nC. 20/400 \nD. 38/400""",
        "gold": "D",
        "steps": [
            "According to the Hardy Weinberg Law, p^2 + 2pq + q^2 = 1, and p + q = 1, where p is the frequency of the dominant allele and q is the frequency of the recessive allele.",
            "The frequency of the recessive allele (q) is the square root of 1/400, which is 0.05.",
            "The frequency of the dominant allele (p) is 1 - q = 0.95.",
            "The frequency of heterozygous individuals is 2pq, which equals 2 x 0.05 x 0.95 = 0.095 or 38/400 when considering the given population size."
        ]
    },
    {
        "prompt": """According to the pressure-flow model of movement of phloem contents, photosynthate movement from source to sink is driven by\n\nOptions:\nA. an ATP-dependent pressure-flow pump \nB. a water-pressure potential gradient \nC. transpiration \nD. apoplastic diffusion""",
        "gold": "B",
        "steps": [
            "It is a gradient in water pressure that induces the movement of phloem content, which refers to answer (B).",
            "The mechanism of movement does not rely on metabolism, which rules out (A).",
            "Transpiration refers to the exhalation of water vapor through plant stomata, and is also not related, which rules out (C).",
            "While the apoplastic pathway is one of two main pathways for water transport in plants, it is not central to the pressure flow model, which rules out (D)."
        ]
    },
    {
        "prompt": """Which of the following contain DNA sequences required for the segregation of chromosomes in mitosis and meiosis?\n\nOptions:\nA. Telomeres \nB. Centromeres \nC. Nucleosomes \nD. Spliceosomes""",
        "gold": "B",
        "steps": [
            "The genetic material in Telomeres is not used, which rules out (A).",
            "Nucleosomes are the repeating subunit that comprises chromatin packed in a cell nucleus, and do not specifically refer to DNA sequences necessary for segregating chromosomes in cell division, which rules out (C).",
            "A spliceosome is a large ribonucleoprotein that removes introns from transcribed pre-mRNA rather than governing chromosome segregation.",
            "Centromeres are directly responsible for segregating chromosomes in cell division."
        ]
    },
    {
        "prompt": """The presence of homologous structures in two different organisms, such as the humerus in the front limb of a human and a bird, indicates that\n\nOptions:\nA. the human and bird are polyphyletic species \nB. a human's and bird's evolution is convergent \nC. the human and bird belong to a clade \nD. the human and bird developed by analogy""",
        "gold": "C",
        "steps": [
            "Polyphyletic species are organisms that are grouped due to having similar characteristics but which do not have a common ancestor. This is not the case for humans and birds, which rules out (A).",
            "Convergent evolution refers to the independent development of similar features in different species at different periods, which is also not the case for humans and birds, which rules out (B).",
            "Analogy refers to the superficial resemblance of structures that have different origins, which is not the case for the human and bird forearms, which rules out (D).",
            "Humans and birds do belong to the same clade - a group of organisms composed of a common ancestor."
        ]
    },
    {
        "prompt": """An expected side effect of creatine supplementation is:\n\nOptions:\nA. muscle weakness. \nB. gain in body mass. \nC. muscle cramps. \nD. loss of electrolytes.""",
        "gold": "B",
        "steps": [
            "Creatine supplementation is a dietary supplement.",
            "Its primary effect is to increase muscle mass and strength.",
            "One of the notable effects of creatine supplementation is a gain in body mass."
        ]
    },
    {
        "prompt": """Which of the following is not a true statement?\n\nOptions:\nA. Muscle glycogen is broken down enzymatically to glucose-1-phosphate \nB. Elite endurance runners have a high proportion of Type I fibres in their leg muscles \nC. Liver glycogen is important in the maintenance of the blood glucose concentration \nD. Insulin promotes glucose uptake by all tissues in the body""",
        "gold": "D",
        "steps": [
            "Muscle glycogen is broken down enzymatically to glucose-1-phosphate, making statement (A) correct.",
            "Elite endurance runners indeed have a high proportion of Type I fibres in their leg muscles, validating statement (B).",
            "Liver glycogen plays a role in maintaining blood glucose concentration, thus statement (C) is accurate.",
            "Insulin does not promote glucose uptake by all tissues in the body. It mainly affects the liver, adipose tissue, and muscle. Notably, tissues in the brain and red blood cells are not influenced by insulin, making statement (D) inaccurate."
        ]
    },
    {
        "prompt": """A high school science teacher fills a 1 liter bottle with pure nitrogen and seals the lid. The pressure is 1.70 atm, and the room temperature is 25°C. Which two variables will both increase the pressure of the system, if all other variables are held constant?\n\nOptions:\nA. Increasing temperature, increasing moles of gas \nB. Increasing temperature, increasing volume \nC. Decreasing volume, decreasing temperature \nD. Decreasing moles of gas, increasing volume""",
        "gold": "A",
        "steps": [
            "The ideal gas law (PV=nRT) indicates that to increase the pressure of a system, the number of moles (n) or temperature (T) needs to increase.",
            "Therefore, both increasing the temperature and increasing the moles of gas would result in an increase in pressure."
        ]
    },
    {
        "prompt": """In a genetic test of a newborn, a rare genetic disorder is found that has X-linked recessive transmission. Which of the following statements is likely true regarding the pedigree of this disorder?\n\nOptions:\nA. All descendants on the maternal side will have the disorder. \nB. Females will be approximately twice as affected as males in this family. \nC. All daughters of an affected male will be affected. \nD. There will be equal distribution of males and females affected.""",
        "gold": "C",
        "steps": [
            "Females have two X chromosomes, while males have one X and one Y chromosome.",
            "If a male is affected by this rare genetic disorder, he will pass the disorder to all his daughters as he can only pass his one X chromosome to them.",
            "Therefore, all daughters of an affected male will inherit the disorder."
        ]
    },
    {
        "prompt": """Glucose is transported into the muscle cell:\n\nOptions:\nA. via protein transporters called GLUT4. \nB. only in the presence of insulin. \nC. via hexokinase. \nD. via monocarbylic acid transporters.""",
        "gold": "A",
        "steps": [
            "Glucose (also known as the blood sugar) is the main sugar found in the human body.",
            "It is transported into the muscle cell via diffusion through protein transporters called GLUT4."
        ]
    },
    {
        "prompt": """3 Cl−(aq) + 4 CrO_4^2−(aq) + 23 H+(aq) → 3 HClO2(aq) + 4 Cr3+(aq) + 10 H2O(l). In the reaction shown above, Cl−(aq) behaves as\n\nOptions:\nA. an acid \nB. a base \nC. a catalyst \nD. a reducing agent""",
        "gold": "D",
        "steps": [
            "A molecule that behaves as a base accepts an H+ ion from another molecule, whereas a molecule that behaves as an acid donates an H+ ion to another molecule.",
            "In the reaction, Cl− does not donate or accept an H+ ion, ruling out options (A) and (B).",
            "A catalyst accelerates a reaction without undergoing a chemical change, but Cl− does undergo a change in this reaction, ruling out option (C).",
            "Cl− carries a negative charge, which it donates in the reaction to form HClO2. This is the behavior of a reducing agent.",
            "Therefore, a reducing agent is the correct answer."
        ]
    },
    {
        "prompt": """Which of the following statements about the lanthanide elements is NOT true?\n\nOptions:\nA. The most common oxidation state for the lanthanide elements is +3. \nB. Lanthanide complexes often have high coordination numbers (> 6). \nC. All of the lanthanide elements react with aqueous acid to liberate hydrogen. \nD. The atomic radii of the lanthanide elements increase across the period from La to Lu.""",
        "gold": "D",
        "steps": [
            "The atomic radii of the lanthanide elements decrease across the period from La to Lu, not increase.",
            "Options (A), (B), and (C) are all true.",
            "Therefore, the statement in option (D) is the only one that is NOT true."
        ]
    },
    {
        "prompt": """Which of the following lists the hydrides of group-14 elements in order of thermal stability, from lowest to highest?\n\nOptions:\nA. PbH4 < SnH4 < GeH4 < SiH4 < CH4 \nB. PbH4 < SnH4 < CH4 < GeH4 < SiH4 \nC. CH4 < SiH4 < GeH4 < SnH4 < PbH4 \nD. CH4 < PbH4 < GeH4 < SnH4 < SiH4""",
        "gold": "A",
        "steps": [
            "The thermal stability of group-14 hydrides decreases as we move from the top of group 14 to the bottom.",
            "The order of elements in the group from top to bottom is C, Si, Ge, Sn, Pb.",
            "Therefore in order of increasing thermal stability we have PbH4, SnH4, GeH4, SiH4, and CH4."
        ]
    },
    {
        "prompt": """Predict the number of lines in the EPR spectrum of a solution of 13C-labelled methyl radical (13CH3•), assuming the lines do not overlap.\n\nOptions:\nA. 4 \nB. 3 \nC. 6 \nD. 24 \nE. 8""",
        "gold": "E",
        "steps": [
            "The electron paramagnetic resonance spectrum will be split by two forms of interactions.",
            "The first is the hyperfine interaction with the 13C (nuclear spin I = 1/2) which will split the spectrum into 2 lines.",
            "This will be further split into 4 lines by the interaction with three equivalent 1H nuclei.",
            "The total number of lines is therefore 2 x 4 = 8."
        ]
    }
]