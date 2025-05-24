document.addEventListener('DOMContentLoaded', function() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const symptomsInput = document.getElementById('symptomsInput');
    const resultsSection = document.getElementById('resultsSection');
    const emergencyWarning = document.getElementById('emergencyWarning');
    const severityProgress = document.getElementById('severityProgress');
    const severityText = document.getElementById('severityText');
    const severityDescription = document.getElementById('severityDescription');
    const severityBadge = document.getElementById('severityBadge');
    const specialistName = document.getElementById('specialistName');
    const specialistDescription = document.getElementById('specialistDescription');
    const specialistIcon = document.getElementById('specialistIcon');
    const urgencyTextEl = document.getElementById('urgencyText'); // Бул элементтин атын өзгөрттүм (мурда severityText менен бирдей болчу)
    const urgencyDescriptionEl = document.getElementById('urgencyDescription');
    const urgencyIconEl = document.getElementById('urgencyIcon');

    // Толук талдоо үчүн жаңы элементтер
    const detailedAnalysisContainer = document.getElementById('detailedAnalysis'); // <--- ЖАҢЫ
    const originalComplaintTextEl = document.getElementById('originalComplaintText');
    const enhancedComplaintTextEl = document.createElement('p'); // <--- ЖАҢЫ (динамикалык түрдө кошулат)
    enhancedComplaintTextEl.className = 'mb-2 text-sm text-gray-400';
    const processedComplaintTextEl = document.getElementById('processedComplaintText');
    const geminiExplanationEl = document.createElement('div'); // <--- ЖАҢЫ
    geminiExplanationEl.className = 'mt-4 prose prose-sm max-w-none text-gray-200';


    const conditionsList = document.getElementById('conditionsList');
    const errorSection = document.getElementById('errorSection');
    const errorMessageEl = document.getElementById('errorMessage');

    // `aiResponseEl` деп өзгөрттүм, анткени бул Gemini'нин жалпы жообу үчүн болчу, эми деталдаштырабыз
    // const aiResponseEl = document.getElementById('aiResponse'); // Бул элемент эми колдонулбайт, ордуна geminiExplanationEl ж.б.

    const backendUrl = 'http://127.0.0.1:5000/predict'; // Flask сервериңиздин дареги

    // Адистер жана шашылыш деңгээлдери боюнча маалыматтар (өзгөргөн жок)
    const specialistsData = {
        'терапевт': { icon: 'fas fa-stethoscope', description: 'Кеңири профилдеги ооруларды аныктоо жана дарылоо боюнча жалпы адис.' },
        'невролог': { icon: 'fas fa-brain', description: 'Нерв системасынын ооруларын, баш ооруну, уйкусуздукту дарылайт.' },
        'кардиолог': { icon: 'fas fa-heartbeat', description: 'Жүрөк жана кан тамыр оорулары боюнча адис.' },
        'гастроэнтеролог': {icon: 'fas fa-pills', description: 'Ашказан-ичеги жолдорунун ооруларына адистешкен дарыгер.' }, // Иконканы туураладым
        'пульмонолог': { icon: 'fas fa-lungs', description: 'Өпкө жана дем алуу жолдорунун оорулары боюнча адис.' },
        'уролог': {icon: 'fas fa-toilet-paper', description: 'Заара чыгаруу системасынын ооруларын дарылайт.' },  // Иконканы туураладым
        'эндокринолог': { icon: 'fas fa-cogs', description: 'Гормоналдык бузулууларды жана зат алмашуу ооруларын дарылайт.' }, // Иконканы туураладым
        'дерматолог': {icon: 'fas fa-hand-sparkles', description: 'Тери, чач жана тырмак оорулары боюнча адис.' }, // Иконканы туураладым
        'офтальмолог': { icon: 'fas fa-eye', description: 'Көз ооруларына адистешкен дарыгер.' },
        'лор': {icon: 'fas fa-head-side-cough', description: 'Кулак, мурун, тамак оорулары боюнча адис (отоларинголог).' }, // Иконканы туураладым
        'хирург': { icon: 'fas fa-band-aid', description: 'Операциялык кийлигишүүнү талап кылган ооруларды дарылоочу адис.' }, // Иконканы туураладым
        'травматолог': { icon: 'fas fa-procedures', description: 'Жаракаттарды, сыныктарды жана таяныч-кыймыл аппаратынын ооруларын дарылайт.' }, // Иконканы туураладым
        'педиатр': { icon: 'fas fa-baby', description: 'Балдардын ден соолугу жана оорулары боюнча адис.' },
        'гинеколог': { icon: 'fas fa-venus', description: 'Аялдардын ден соолугу жана репродуктивдик системасы боюнча адис.' },
        'стоматолог': { icon: 'fas fa-tooth', description: 'Тиш жана ооз көңдөйүнүн ооруларын дарылоочу дарыгер.' },
        'психотерапевт': { icon: 'fas fa-comment-medical', description: 'Психикалык ден соолук жана эмоционалдык маселелер боюнча адис.' },
        'нефролог': { icon: 'fas fa-kidneys', description: 'Бөйрөк оорулары боюнча адис.' },
        'аллерголог': { icon: 'fas fa-allergies', description: 'Аллергиялык реакциялар жана оорулар боюнча адис.' },
        'ревматолог': { icon: 'fas fa-joint', description: 'Муун жана тутумдаштыргыч ткандардын оорулары боюнча адис.'},
        'флеболог': { icon: 'fas fa-water', description: 'Вена (кан тамыр) оорулары боюнча адис.'}, // Иконканы туураладым
        'пульмонолог': { icon: 'fas fa-lungs', description: 'Өпкө жана дем алуу жолдорунун оорулары боюнча адис.' },
        'проктолог': { icon: 'fas fa-poop', description: 'Көтөн чучук жана ага жакын органдардын оорулары боюнча адис.'},
        'не определено': { icon: 'fas fa-question-circle', description: 'Дарыгер так аныкталган жок. Сураныч, белгилериңизди кененирээк жазыңыз.' },
        'хирург/травматолог': { icon: 'fas fa-procedures', description: 'Жаракаттарды, сыныктарды жана операцияны талап кылган кырдаалдарда жардам берет.' },
         'педиатр/дерматолог': { icon: 'fas fa-baby', description: 'Балдардын тери оорулары боюнча адис.' },
         'терапевт/невролог': { icon: 'fas fa-stethoscope', description: 'Жалпы терапиялык жана нерв системасы маселелери боюнча жардам берет.' },
         'хирург/флеболог': { icon: 'fas fa-band-aid', description: 'Вена оорулары жана хирургиялык кийлигишүүлөр боюнча адис.' },
         'педиатр/хирург': { icon: 'fas fa-baby', description: 'Балдар хирургиясы боюнча адис.' },
         'стоматолог/невролог': { icon: 'fas fa-tooth', description: 'Тиш жана жаак-бет аймагындагы неврологиялык маселелер боюнча адис.' },
         'лор/терапевт': { icon: 'fas fa-head-side-cough', description: 'Кулак-мурун-тамак жана жалпы терапиялык маселелер боюнча адис.' },
         'терапевт/проктолог': { icon: 'fas fa-stethoscope', description: 'Жалпы терапиялык жана проктологиялык маселелер боюнча адис.' },
         'невролог/хирург': { icon: 'fas fa-brain', description: 'Нерв системасынын хирургиялык дарылоону талап кылган оорулары боюнча адис.' },
         'кардиолог/терапевт': { icon: 'fas fa-heartbeat', description: 'Жүрөк-кан тамыр жана жалпы терапиялык маселелер боюнча адис.' },
         'невролог/травматолог': { icon: 'fas fa-brain', description: 'Нерв системасынын жаракаттары жана травматологиялык маселелер боюнча адис.' },
         'гастроэнтеролог/терапевт': { icon: 'fas fa-pills', description: 'Ашказан-ичеги жана жалпы терапиялык маселелер боюнча адис.' }
    };

    const urgencyMapping = {
        'Красный': { key: 'immediate', severityScore: 95, badgeColor: 'bg-red-600 text-red-100', text: 'Критикалык' },
        'Оранжевый': { key: 'urgent', severityScore: 75, badgeColor: 'bg-orange-600 text-orange-100', text: 'Жогорку' },
        'Желтый': { key: 'soon', severityScore: 50, badgeColor: 'bg-yellow-600 text-yellow-100', text: 'Орточо' },
        'Зеленый': { key: 'routine', severityScore: 25, badgeColor: 'bg-green-600 text-green-100', text: 'Төмөн' },
        'Аныкталган жок': { key: 'unknown', severityScore: 0, badgeColor: 'bg-gray-600 text-gray-100', text: 'Белгисиз' }, // <--- ӨЗГӨРТҮЛДҮ
        'Не определено': { key: 'unknown', severityScore: 0, badgeColor: 'bg-gray-600 text-gray-100', text: 'Белгисиз' } // <--- Кошумча вариант
    };

   const urgencyLevelsDetails = {
        'immediate': {
            text: 'Дароо жардам',
            description: 'Өмүргө коркунуч туулушу мүмкүн. Тез арада медициналык жардам талап кылынат!',
            icon: 'fas fa-ambulance text-red-500'
        },
        'urgent': {
            text: 'Шашылыш',
            description: '24 сааттын ичинде дарыгерге кайрылуу сунушталат.',
            icon: 'fas fa-exclamation-triangle text-orange-500'
        },
        'soon': {
            text: 'Жакын арада',
            description: 'Жакынкы күндөрү дарыгерге кайрылуу пландаштырылышы керек.',
            icon: 'fas fa-clock text-yellow-500'
        },
        'routine': {
            text: 'Пландуу',
            description: 'Пландуу түрдө дарыгерге кайрылсаңыз болот.',
            icon: 'fas fa-calendar-check text-green-500'
        },
        'unknown': {
            text: 'Белгисиз',
            description: 'Шашылыш деңгээли аныкталган жок. Белгилериңизди толугураак жазыңыз.',
            icon: 'fas fa-question-circle text-gray-500'
        }
    };


    // Бул демо маалыматтар, Gemini'ден чыныгы маалыматтарды алууга болот
    const sampleConditions = [
        // Бул жерди бош калтырсаңыз болот же серверден чыныгы маалымат келбесе, көрсөтүү үчүн калтырыңыз
    ];


    analyzeBtn.addEventListener('click', async function() {
        const symptoms = symptomsInput.value.trim();

        if (!symptoms) {
            showError('Сураныч, белгилериңизди жазыңыз.');
            return;
        }

        setLoadingState(true);

        try {
            const response = await fetch(backendUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ complaint: symptoms }), // Python күткөн 'complaint' ачкычы
            });

            if (!response.ok) {
                let errorMsg = `Сервер катасы: ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.error_ky || errorData.error || errorMsg; // <--- ӨЗГӨРТҮЛДҮ
                } catch (e) { /* JSON парсинг катасы болсо, унчукпай коёбуз */ }
                throw new Error(errorMsg);
            }

            const data = await response.json();
            resultsSection.classList.remove('hidden');
            errorSection.classList.add('hidden'); // Ийгиликтүү болсо, катаны жашыруу

            // --- Натыйжаларды көрсөтүү ---

            // Шашылыш деңгээли
            const backendUrgencyKey = data.predicted_urgency_ky || 'Аныкталган жок'; // <--- ӨЗГӨРТҮЛДҮ
            const urgencyInfo = urgencyMapping[backendUrgencyKey] || urgencyMapping['Аныкталган жок'];

            severityProgress.style.width = urgencyInfo.severityScore + '%';
            severityText.textContent = urgencyInfo.text; // Бул badge үчүн
            severityBadge.className = `px-4 py-1 rounded-full text-sm font-medium ${urgencyInfo.badgeColor}`;
            if (urgencyInfo.severityScore > 0) { // <--- ӨЗГӨРТҮЛДҮ: Эгер белгисиз болсо, badge'ди көрсөтпөө
                 severityBadge.classList.remove('hidden');
            } else {
                 severityBadge.classList.add('hidden');
            }


            const urgencyDetails = urgencyLevelsDetails[urgencyInfo.key];
            urgencyTextEl.textContent = urgencyDetails.text; // Бул өзүнчө блок үчүн
            urgencyDescriptionEl.textContent = urgencyDetails.description;
            urgencyIconEl.className = `${urgencyDetails.icon} text-4xl mb-2`;


            if (urgencyInfo.key === 'immediate') {
                emergencyWarning.classList.remove('hidden');
            } else {
                emergencyWarning.classList.add('hidden');
            }

            // Сунушталган адис
            const backendSpecialistRaw = data.predicted_specialist_ky || 'Аныкталган жок'; // <--- ӨЗГӨРТҮЛДҮ
            const backendSpecialistLC = backendSpecialistRaw.toLowerCase(); // Салыштыруу үчүн кичине тамгага
            const specialistInfo = specialistsData[backendSpecialistLC] || specialistsData['не определено'];

            let specialistDisplayName = backendSpecialistRaw.charAt(0).toUpperCase() + backendSpecialistRaw.slice(1);
            if (backendSpecialistRaw === 'Аныкталган жок') specialistDisplayName = 'Аныкталган жок';


            specialistName.textContent = specialistDisplayName;
            specialistDescription.textContent = specialistInfo.description;
            specialistIcon.className = `${specialistInfo.icon} text-4xl mb-2`;


            // Толук талдоо бөлүмү
            detailedAnalysisContainer.innerHTML = ''; // Эски мазмунду тазалоо

            originalComplaintTextEl.textContent = `Баштапкы арыз: ${data.complaint_original_ky || symptoms}`; // <--- ӨЗГӨРТҮЛДҮ
            detailedAnalysisContainer.appendChild(originalComplaintTextEl);

            if (data.complaint_enhanced_ky && data.complaint_enhanced_ky !== "Жакшыртуу колдонулган жок же өзгөргөн жок") { // <--- ӨЗГӨРТҮЛДҮ
                enhancedComplaintTextEl.textContent = `AI жакшырткан арыз: ${data.complaint_enhanced_ky}`;
                detailedAnalysisContainer.appendChild(enhancedComplaintTextEl);
            }

            processedComplaintTextEl.textContent = `Модель үчүн иштетилген арыз: ${data.complaint_processed_for_model_ky || 'маалымат жок'}`; // <--- ӨЗГӨРТҮЛДҮ
            detailedAnalysisContainer.appendChild(processedComplaintTextEl);

            // Gemini'ден алынган түшүндүрмөлөр
            geminiExplanationEl.innerHTML = ''; // Тазалоо
            if (data.gemini_explanation_ky && data.gemini_explanation_ky !== "Түшүндүрмө түзүлгөн жок.") { // <--- ӨЗГӨРТҮЛДҮ
                const explanationTitle = document.createElement('h4');
                explanationTitle.className = 'font-semibold text-cyan-400 mt-3 mb-1';
                explanationTitle.textContent = 'AI Түшүндүрмөсү:';
                geminiExplanationEl.appendChild(explanationTitle);
                const explanationP = document.createElement('p');
                explanationP.textContent = data.gemini_explanation_ky;
                geminiExplanationEl.appendChild(explanationP);
            }

            if (data.gemini_follow_up_questions_ky && data.gemini_follow_up_questions_ky.length > 0) { // <--- ӨЗГӨРТҮЛДҮ
                const questionsTitle = document.createElement('h4');
                questionsTitle.className = 'font-semibold text-cyan-400 mt-3 mb-1';
                questionsTitle.textContent = 'Тактоочу суроолор:';
                geminiExplanationEl.appendChild(questionsTitle);
                const questionsUl = document.createElement('ul');
                questionsUl.className = 'list-disc list-inside ml-4';
                data.gemini_follow_up_questions_ky.forEach(q => {
                    const li = document.createElement('li');
                    li.textContent = q;
                    questionsUl.appendChild(li);
                });
                geminiExplanationEl.appendChild(questionsUl);
            }

            if (data.gemini_general_advice_ky) { // <--- ӨЗГӨРТҮЛДҮ
                const adviceTitle = document.createElement('h4');
                adviceTitle.className = 'font-semibold text-cyan-400 mt-3 mb-1';
                adviceTitle.textContent = 'Жалпы кеңеш:';
                geminiExplanationEl.appendChild(adviceTitle);
                const adviceP = document.createElement('p');
                adviceP.textContent = data.gemini_general_advice_ky;
                geminiExplanationEl.appendChild(adviceP);
            }
            detailedAnalysisContainer.appendChild(geminiExplanationEl);


            // "Ыктымалдуу абалдар" (демо)
            conditionsList.innerHTML = ''; // Эскисин тазалоо
            // Бул жерде серверден чыныгы ыктымалдуу абалдарды алса болот, азырынча үлгү
            if (sampleConditions.length > 0) {
                 sampleConditions.forEach(condition => {
                    // ... (сиздин мурунку код бул жерде, өзгөрүүсүз)
                });
            } else {
                const noConditionsMsg = document.createElement('p');
                noConditionsMsg.className = 'text-sm text-gray-400';
                noConditionsMsg.textContent = 'Азырынча ыктымалдуу абалдар боюнча маалымат жок.';
                conditionsList.appendChild(noConditionsMsg);
            }


        } catch (error) {
            console.error('Талдоо учурунда ката кетти:', error);
            showError(`Ката: ${error.message}. Сураныч, кайталап көрүңүз же сервер иштеп жатканын текшериңиз.`);
            resultsSection.classList.add('hidden');
        } finally {
            setLoadingState(false);
        }
    });

    function setLoadingState(isLoading) {
        analyzeBtn.disabled = isLoading;
        if (isLoading) {
            analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Талдап жатат...';
        } else {
            analyzeBtn.innerHTML = '<i class="fas fa-redo mr-2"></i> Кайра талдоо';
        }
    }

    function showError(message) {
        errorMessageEl.textContent = message;
        errorSection.classList.remove('hidden');
    }

    // typeWriter функциясы калды, бирок эми түз колдонулбайт, анткени маалымат дароо чыгат
    // Эгер "жазып жаткан" эффект керек болсо, кайра кошсоңуз болот
    // function typeWriter(text, element, speed = 20, i = 0) { ... }
});