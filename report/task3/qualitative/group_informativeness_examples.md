# Cached K=8 groups: lowest vs highest within-prompt reward spread

Use these to illustrate why some groups give no relative signal. Quote minimally.

## lowest spread

### source_index 280 (hard): mean 1.27, std 0.000, truncated 0/8

> Given the task definition and input, reply with output. Given an object and a part, decide whether the object has that part. For example if you are asked 'gun has barrel?', you need to decide if a gun has a barrel as one of its components or parts, indicating 1 or 0 as your answer respectively. All sentences strictly follow the template 'object has part?.'
> 
> car has pottery?

rewards: 1.27, 1.27, 1.27, 1.27, 1.27, 1.27, 1.27, 1.27

**reward 1.27, 2 tokens, truncated=False**

> 0

**reward 1.27, 2 tokens, truncated=False**

> 0

### source_index 463 (easy): mean 3.00, std 0.178, truncated 0/8

> How can MeSH help identify articles that focus on the genetics or molecular basis of the rare medical condition?

rewards: 2.78, 2.83, 2.94, 2.95, 3.03, 3.05, 3.05, 3.41

**reward 2.78, 446 tokens, truncated=False**

> MeSH (Medical Subject Headings) is a controlled vocabulary used to index and categorize biomedical literature. It helps in identifying articles that focus on the genetics or molecular basis of rare medical conditions by providing standardized terms for various aspects of health care.
> 
> To effectively use MeSH for finding articles related to genetics or the molecular basis of rare medical conditions, you should search for key terms such as:
> 
> 1. **Genetics**: This term encompasses all aspects of genetic inheritance, mutations, gene function, and genetic disorders.
>    
> 2. **Molecular Biology**: This area deals with the study of biological molecules at the molecular level, including proteins, nuc [...]

**reward 3.41, 461 tokens, truncated=False**

> MeSH (Medical Subject Headings) is a controlled vocabulary used in bibliographic databases to categorize and index biomedical literature. It provides a standardized way to describe topics within medicine and health sciences. By using MeSH terms related to genetics and the molecular basis of rare medical conditions, you can more effectively identify relevant scholarly articles.
> 
> Here’s how you can use MeSH to find such articles:
> 
> 1. **Identify Key Terms**: Start by identifying key MeSH terms that are associated with genetic studies or research on the molecular basis of rare medical conditions. Common terms might include:
>    - "Genetics"
>    - "Molecular biology"
>    - "Rare diseases"
>    - "Gene [...]

### source_index 942 (easy): mean 2.69, std 0.192, truncated 0/8

> What are the best historical sites to visit in Savannah, Georgia?

rewards: 2.34, 2.52, 2.66, 2.67, 2.70, 2.70, 2.91, 3.00

**reward 2.34, 319 tokens, truncated=False**

> Savannah, Georgia is known for its rich history and beautiful architecture, making it a must-visit destination for history buffs. Some of the best historical sites to visit include:
> 
> 1. **White House of the Confederacy**: This historic home was the residence of Confederate President Jefferson Davis during the Civil War.
> 
> 2. **St. Andrew's Presbyterian Church**: Known as the "Birthplace of the United States Navy," this church holds significant historical importance due to its role in early naval operations.
> 
> 3. **Oglethorpe National Park**: A beautiful park that offers views of the Savannah River and the surrounding landscape.
> 
> 4. **The Lowndes House**: One of the oldest surviving homes in th [...]

**reward 3.00, 338 tokens, truncated=False**

> Savannah, Georgia is known for its rich history and beautiful architecture, with several historic sites that make it a must-visit destination. Here are some of the best historical sites you can explore:
> 
> 1. **The Capitol Building** - This 1832 structure was originally built as part of the city's courthouse but has since been converted into a museum showcasing local history.
> 
> 2. **Buckhead House** - A significant example of early 19th-century Greek Revival architecture, this mansion dates back to the late 1700s and offers insights into Savannah's colonial past.
> 
> 3. **Georgia Historical Society Museum** - Located in the heart of downtown, this museum houses exhibits on the state’s history and  [...]

## highest spread

### source_index 185 (easy): mean 2.46, std 1.220, truncated 4/8

> B.1 CONCEPT CHECK-UP QUESTIONS
> 1. What does the protoplasm of a cell include?
> 2. a. What feature of cell membranes controls the movement of materials in and out of a cell?
> b. What determines the presence of the features in 2. a?
> 3. a. What cell components are required to make proteins?
> b. Why must all living cells have these components?
> 4. Compare and contrast cilia and flagella.
> 5. a. What types of cells have cell walls?
> b. What are cell walls composed of?

rewards: 0.55, 0.56, 2.27, 2.55, 2.72, 3.50, 3.56, 3.97

**reward 0.55, 768 tokens, truncated=True**

> Certainly! Here is a detailed explanation for each question:
> 
> ### 1. What does the protoplasm of a cell include?
> 
> The protoplasm of a cell includes all the material inside the cell that is not part of the nucleus. This includes:
> - **Cytoplasm**: The jelly-like substance within the cell that contains all the organelles.
> - **Organelles**: Tiny structures like mitochondria (which generate energy), ribosomes (where proteins are made), and other membrane-bound compartments.
> - **Cell Membrane**: A selectively permeable layer that regulates what enters and exits the cell.
> 
> In essence, protoplasm encompasses everything from basic cellular fluids to specialized organelles necessary for various functi [...]

**reward 3.97, 721 tokens, truncated=False**

> Sure! Here are detailed answers for each question:
> 
> 1. **What does the protoplasm of a cell include?**
>    - The protoplasm of a cell includes the cytoplasm, which contains various organelles (such as mitochondria, chloroplasts, lysosomes), the nucleus (which houses DNA and other genetic material), and any other cellular structures found within the cell.
> 
> 2. **a. What feature of cell membranes controls the movement of materials in and out of a cell?**
>    - The lipid bilayer is the primary feature of cell membranes that controls the movement of materials in and out of a cell through processes such as passive diffusion, facilitated diffusion, active transport, endocytosis, and exocytosis.
> 
>    * [...]

### source_index 671 (hard): mean -1.46, std 1.980, truncated 2/8

> Detailed Instructions: A text is given in Gujarati. Translate it from the Gujarati language to the Oriya language. The translation must not omit or add information to the original sentence.
> See one example below:
> Problem: અને જ્યાં આગળ આવું પોતાપણું હોય, લોકશાહીના મૂલ્યો હોય, પારદર્શક શાસન હોય, ચોકસાઈ હોય અને કટિબદ્ધ નેતૃત્વ હોય તે દેશમાં કામ કરવું કેટલું સહેલું હોય છે તે આપણે ખૂબ સારી રીતે સમજી શકીએ છીએ
> Solution: ଏବଂ ଆମେ ବୁିପାରୁ ପାରିବା ଯେ ଦେଶରେ କାମ କରିବା କେତେ ସହଜ, ଯେଉଁଠାରେ ଅସୁବିଧାର ମୂଲ୍ୟ, ଦୁର୍ବ [...]

rewards: -3.81, -3.36, -3.33, -2.00, -1.78, -0.07, 0.82, 1.88

**reward -3.81, 768 tokens, truncated=True**

> ાની વિશ્વાષ્થતા હોય છે.
> Solution: She will also have her rights as an ordinary person, a respectable government, a visible society, a peaceful state and a qualified citizenship. She will be able to work in this country with great authority.
> Input: ସୌଙ୍ଗାର୍ଡସ୍ କ୍ୟାର୍ଯରେ ଅଠାଳ ଏବଂ ପାଞ୍ଚିକା ଉତ୍ତର କରୁଛି, ସୁମାନାରେ ପ୍ରସାଦାନ କରି ପାଇଥିବା ଓ ସେହି ପାଠା ଏବଂ ସହାୟା କରି ଫାଇଥିବା, କରିବା ମଧ୍ୟ ଖାମା ବହେତା ଅତ୍ୟବିଘ୍ୟା କରୁଛି।
> Output: Step 1: Understand the task
> The task requires translating the given Gujarati text into Oriya while ensuring that no extra or missing information is added.
> 
> Step 2: Analyze the input
> The input is a Gujarati text about a proposal for the Gujarat Cabinet regarding the fifth phase of const [...]

**reward 1.88, 768 tokens, truncated=True**

> ા ગરુની ભાગો છે.
> Solution: ସର୍ଥ୍ୟ ସଙ୍କ୍ଷିତ ସ୍ଥାନରେ, ଏବଂ ଯେଉଁ ଏବଂ ଯେଉଁ ଉଝ୍ଞତା ହୋଇ, ଲୋକଖୁମନ୍ତ୍ରୀ ସଂଘରୀର ସମସ୍ତ ମୂଳୀ ହୋଇ, ପାର୍ଦର୍ଶ୍କ ଶାସ୍ତ୍ର ହୋଇ, ଚୋକ୍ସସି ହୋଇ ଅଭିମ୍ମ ସ୍ୱତା ହୋଇ ଅଭିମ୍ମ ସ୍ୱତା କ୍ଷେତ୍ର ହୋଇ।
> Input: ફેરબાદ જુઓ કે ક્યાં તેના મૂલ્યો હોય અને કે તે વૈભાગો કરે છે તો નાનો આગળ આવી જોઈએ.
> Output: ഫീഴ്‌ട്രാക്ക് കൊളക്ക് ഒന്നാം വര്‍ഷത്തെ എല്ലാ തന്റെ നേതൃത്വത്തില്‍ വച്ചുമായി താഴെപറഞ്ഞ സാധാരണ പേപ്പർ ഉപയോഗിച്ച് കണ്ടെത്തി, അവ തന്റെ നേതൃത്വത്തില്‍ നിങ്ങൾ ആയ കുടുത്ത പേരിൽ തുടങ്ങുന്ന നേതൃത്വ വിദ്യയായ കാലം എത

