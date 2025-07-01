# CambioC.P.Final
Este proyecto final llamado "Cambio.C.P.Final", va a ser utilizado por las personas para que puedan ver como está el lugar, cerca de sus casas, o el lugar a donde vayan (Flora, Fauna, Animales, y cosas así).
## ¿Qué va a contener?
Mi proyecto va a contener:
  1. Un modelo entrenado (En Google Techable Machine):
  - Va a tener que describir y analizar de que se tratan las imagenes. Por ejemplo diferenciar si el entorno tiene buena flora y fauna, si hay mucha contaminación.
  - Deben ser unos 4 o hasta 6 modelos entrenados los que debo tener. Ejemplos de modelos:
    - Lugar con mucha o poca contaminación
    - Glaciares derretiendose o Glaciares normales
    - Bosque contaminado por CO2 o bosque normal
    - Y más ideas que se me irán ocurriendo con el paso del tiempo.
  - Cada uno de los modelos entrenados va a tener su nombre para que lo puedas utilizar en DISCORD. Algunos ejemplos son:
    - Para saber si en un lugar hay mucha contaminación. Eco_Lugar
    - Para los Glaciares. Glaciares
    - Para los bosques. Eco_Bosques
  2. Programar en VisualStudioCode:
  - Con códigos como los ya mencionados, y para hablar con el bot hay que colocar "%" al principio de la oración.
  - Colocar al bot un botón, para subir y descargar una imagen para que el modelo lo analice.
  - Colocar que analice la imagen enviada durante 10 segundos y que te envíe que porcentaje de cercanía a la limpieza está tu imagen o que porcentaje de suciedad y de cambio ambiental tiene tu imagen. Y mandar un mensaje. 
  3. Utilizar lo ya mencionado (Discord):
    - Colocar "%", para hablar
    - Utilizar para hablar los ya mencionados códigos de Eco_Lugar, Glaciares, Eco_Bosque y todo lo demás.
    - En el botón del bot colocar la imagen deseada y colocar el modelo en el que quieres que tu imagen sea evaluada.
  4.  Mensaje para mejorar:
    - Después de analizar la imagen en el lugar deseado, te va a mandar el porcentaje de limpieza de tu imagen o el porcentaje de contaminacion, o cercanía a ella está tu imagen.
    - Después te va a enviar un mensaje que diga "¿Deseas mejorar ese porcentaje y acercarte a la limpieza?".
    - Si respondes sí:
      - Te va a enviar un mensaje según el porcentaje de tu imagen, ese mensaje te va a decir sobre como mejorar, que hacer, que hechar, que limpiar y cosas así.
    - Si respondes no:
      - Te enviara un mensaje diciendo "OK, gracias por enviar tu imagen. Hemos finalizado."
  5.(OPCIONAL/SI SOBRA TIEMPO/ O ENTIENDES COMO HACER) Try y Except:
    - Si estás colocando una imagen de un glaciar en Glaciares, te va a decir "Muy bien, analizando porcentaje."
    - Pero, como siempre hay personas que bobean con esto, si alguien coloca una foto de un glaciar en Eco_Bosques o en algún otro que no sea Glaciares. Se le va a enviar un mensaje que diga "ERROR la imagen no tiene relevancia con el modelo, por favor coloca la imagen en otro modelo."





import discord
from discord.ext import commands
import os
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import requests
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv('TOKEN')


COMMAND_PREFIX = '!'
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents)


MODEL_PATHS = {
    'eco_bosque': {
        'path': 'modelseco_bosque_model.h5',
        'labels': 'modelseco_bosque_labels.txt',
        'model': None,
        'labels_list': [],
        'good_class_name': 'Bosques limpios'
    },
    'eco_ciudad': {
        'path': 'modelseco_ciudad_model.h5',
        'labels': 'modelseco_ciudad_labels.txt',
        'model': None,
        'labels_list': [],
        'good_class_name': 'Ciudades Limpias'
    },
    'glaciares': {
        'path': 'modelsglaciares_model.h5',
        'labels': 'modelsglaciares_labels.txt',
        'model': None,
        'labels_list': [],
        'good_class_name': 'Glaciares normales'
    }
}

IMAGE_INPUT_SIZE = (224, 224)


MESSAGES = {
    'eco_bosque': {
        '0-25': "Parece que la imagen de tu bosque podría mejorar mucho. Intenta que la imagen se centre más en árboles, vegetación densa y, si es posible, que capture la vitalidad del ecosistema forestal. Evita elementos que no sean del bosque.",
        '26-50': "Tu imagen de bosque tiene potencial, ¡pero podemos hacerla brillar más! Intenta incluir más detalles de la flora, la luz que se filtra entre los árboles o incluso la fauna. Evita el cielo excesivo o elementos extraños.",
        '51-75': "¡Genial! Tu imagen ya representa bien un bosque. Para mejorar aún más, enfócate en la composición, la iluminación natural y la riqueza de detalles de la biodiversidad forestal. ¡Casi lo tienes!",
        '76-100': "¡Felicidades! Tu imagen es un excelente ejemplo de un bosque saludable. ¡El modelo lo ha reconocido perfectamente! Sigue así con tus capturas."
    },
    'eco_ciudad': {
        '0-25': "Hmm, esta imagen de ciudad no parece muy limpia. Para mejorar, busca tomas con arquitectura clara, calles ordenadas, ausencia de basura visible y, si es posible, espacios verdes bien cuidados. Evita multitudes desordenadas o grafitis excesivos.",
        '26-50': "Tu imagen de ciudad tiene algunos elementos positivos, pero aún hay margen de mejora. Intenta capturar una sensación de orden, limpieza en la calle, o infraestructura moderna. La iluminación también puede hacer una gran diferencia.",
        '51-75': "¡Muy bien! Tu imagen de ciudad ya se ve bastante limpia. Para subir de nivel, busca ángulos que resalten la estética urbana, la limpieza de las aceras o la presencia de áreas verdes bien mantenidas. ¡Casi una postal!",
        '76-100': "¡Impresionante! Tu imagen es un claro ejemplo de una ciudad limpia y bien cuidada. ¡El modelo la ha clasificado de manera excelente! Continúa capturando esos detalles urbanos."
    },
    'glaciares': {
        '0-25': "Parece que esta imagen de glaciar no muestra su estado óptimo. Para mejorar, enfócate en capturar grandes masas de hielo, nieve compacta y, si es posible, tonos azules intensos que indican hielo antiguo y denso. Evita el agua en exceso o la tierra expuesta.",
        '26-50': "Tu imagen de glaciar tiene algunos elementos, pero podemos hacerla más representativa de un glaciar saludable. Busca ángulos que muestren la inmensidad del hielo, la falta de deshielo visible o la pureza de la nieve. La luz polar puede ayudar mucho.",
        '51-75': "¡Excelente! Tu imagen ya se ve como un glaciar en buen estado. Para mejorar, busca la majestuosidad de la formación de hielo, la pureza del blanco y azul, y la ausencia de signos de derretimiento. ¡Casi perfecta!",
        '76-100': "¡Magnífico! Tu imagen es una representación perfecta de un glaciar saludable. El modelo ha detectado su estado óptimo. ¡Sigue documentando estos paisajes increíbles!"
    }
}


async def load_tm_model(model_name):
    """Carga un modelo de Teachable Machine y sus etiquetas."""
    try:
        print(f"Cargando modelo '{model_name}' desde {MODEL_PATHS[model_name]['path']}...")
        MODEL_PATHS[model_name]['model'] = tf.keras.models.load_model(MODEL_PATHS[model_name]['path'], compile=False)

        print(f"Cargando etiquetas para '{model_name}' desde {MODEL_PATHS[model_name]['labels']}...")
        with open(MODEL_PATHS[model_name]['labels'], 'r', encoding='utf-8') as f:
            MODEL_PATHS[model_name]['labels_list'] = [line.strip() for line in f]

        print(f"Modelo '{model_name}' y etiquetas cargados exitosamente.")
        return True
    except Exception as e:
        print(f"Error al cargar el modelo '{model_name}': {e}")
        return False

async def preprocess_image(image_bytes):
    """Preprocesa la imagen para que sea compatible con los modelos de TM."""
    try:
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image = ImageOps.fit(image, IMAGE_INPUT_SIZE, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        return np.expand_dims(normalized_image_array, axis=0)
    except Exception as e:
        print(f"Error en el preprocesamiento de la imagen: {e}")
        return None

async def _send_details_and_advice(ctx, model_name_key, all_probabilities, good_class_name):
    """Envía al usuario los porcentajes de todas las clases y el consejo de mejora."""
    detail_message = f"Aquí tienes los porcentajes de todas las clases para tu imagen con el modelo **{model_name_key.replace('_', ' ').title()}**:\n"
    for label, prob in sorted(all_probabilities.items(), key=lambda item: item[1], reverse=True):
        detail_message += f"**{label}: {prob:.2f}%**\n"

    await ctx.send(detail_message)

    good_class_percentage = all_probabilities.get(good_class_name, 0.0)

    advice_message = get_improvement_message(model_name_key, good_class_percentage)
    await ctx.send(f"\n**Consejo para mejorar tu '{good_class_name}':**\n{advice_message}")


async def predict_with_model(ctx, model_info, image_url):
    """Descarga la imagen, la procesa y hace una predicción con el modelo dado."""
    model = model_info['model']
    labels = model_info['labels_list']
    model_name_key = next(key for key, value in MODEL_PATHS.items() if value == model_info)
    model_display_name = model_info['path'].split('/')[-1].replace('_model.h5', '').replace('_folder', '')

    if not model:
        await ctx.send(f"El modelo '{model_display_name}' no está cargado o hubo un error. Por favor, inténtalo más tarde.")
        return

    try:
        await ctx.send(f"Procesando imagen con el modelo '{model_display_name}'...")
        response = requests.get(image_url)
        image_data = await preprocess_image(response.content)

        if image_data is None:
            await ctx.send("No pude procesar la imagen adjunta. Asegúrate de que sea una imagen válida.")
            return

        prediction = model.predict(image_data)
        
        all_probabilities = {}
        for i, prob in enumerate(prediction[0]):
            label = labels[i]
            clean_label = label[2:] if label[0].isdigit() and label[1] == ' ' else label
            all_probabilities[clean_label] = prob * 100

        predicted_class_index = np.argmax(prediction[0])
        predicted_probability = prediction[0][predicted_class_index]

        reply_message = f"Resultados del modelo **{model_display_name}** para tu imagen:\n"
        if predicted_class_index < len(labels):
            predicted_label = labels[predicted_class_index]
            clean_label = predicted_label[2:] if predicted_label[0].isdigit() and predicted_label[1] == ' ' else predicted_label
            reply_message += f"**{clean_label}: {(predicted_probability * 100):.2f}%**\n"
        else:
            reply_message += f"Clase desconocida {predicted_class_index}: {(predicted_probability * 100):.2f}%\n"
        
        reply_message += f"\nSi deseas ver los porcentajes de todas las clases y obtener consejos para mejorar tu '{model_info['good_class_name']}', usa el comando `{COMMAND_PREFIX}detalles`."
        
        await ctx.send(reply_message)
        
        bot.last_prediction_data[ctx.author.id] = { 
            'channel_id': ctx.channel.id, 
            'model_name_key': model_name_key,
            'all_probabilities': all_probabilities,
            'good_class_name': model_info['good_class_name']
        }

    except Exception as e:
        print(f"Error en la predicción con el modelo {model_display_name}: {e}")
        await ctx.send(f"Hubo un error al intentar la predicción con el modelo {model_display_name}. Detalles: {e}")

def get_improvement_message(model_name_key, percentage):
    messages_for_model = MESSAGES.get(model_name_key, {})

    if percentage >= 76:
        return messages_for_model.get('76-100', "¡Excelente trabajo!")
    elif percentage >= 51:
        return messages_for_model.get('51-75', "¡Vas muy bien, sigue así!")
    elif percentage >= 26:
        return messages_for_model.get('26-50', "Podemos mejorar, ¡ánimo!")
    else: 
        return messages_for_model.get('0-25', "No te rindas, ¡hay mucho por aprender!")

# --- Eventos del Bot ---
@bot.event
async def on_ready():
    """Se ejecuta cuando el bot se conecta a Discord."""
    print(f'{bot.user} se ha conectado a Discord!')
    print("Cargando todos los modelos Teachable Machine...")

    success_all = True
    for model_key in MODEL_PATHS:
        success = await load_tm_model(model_key)
        if not success:
            print(f"Advertencia: El modelo '{model_key}' no pudo cargarse.")
            success_all = False

    if success_all:
        print("Todos los modelos cargados exitosamente.")
    else:
        print("Carga de algunos modelos falló. Revisa los logs.")
    
    bot.last_prediction_data = {}


# --- Comandos del Bot ---


@bot.command(name='ayuda') 
async def custom_help_command(ctx):
    """Muestra una guía completa de cómo usar el bot."""
    help_message = (
        f"¡Hola {ctx.author.display_name}! Soy tu bot ambiental. ¡Estoy aquí para ayudarte a analizar imágenes y aprender sobre el medio ambiente!\n\n"
        f"Usa el prefijo `{COMMAND_PREFIX}` antes de cada comando.\n\n"
        "**Comandos para analizar imágenes:**\n"
        f"• `{COMMAND_PREFIX}ecobosque`: Envía una imagen de un bosque. Yo la analizaré para ver su estado.\n"
        f"• `{COMMAND_PREFIX}ecociudad`: Envía una imagen de una ciudad. Yo la revisaré para evaluar su limpieza.\n"
        f"• `{COMMAND_PREFIX}glaciares`: Envía una imagen de un glaciar. Yo la examinaré para determinar su condición.\n"
        "*(Recuerda adjuntar la imagen al usar estos comandos)*\n\n"
        "**Comando para detalles y consejos:**\n"
        f"• `{COMMAND_PREFIX}detalles`: Después de que analice tu imagen con cualquiera de los comandos anteriores, puedes usar este comando para ver un desglose completo de los porcentajes de cada categoría y recibir consejos personalizados sobre cómo mejorar tus resultados."
    )
    await ctx.send(help_message)

@bot.command(name='help_tm')
async def help_command_tm(ctx):
    """Alias para el comando de ayuda principal."""
    await custom_help_command(ctx) 


@bot.command(name='ecobosque')
async def ecobosque_command(ctx):
    """Analiza una imagen con el modelo Eco_Bosque."""
    if not ctx.message.attachments:
        await ctx.send(f"Por favor, adjunta una imagen con el comando `{COMMAND_PREFIX}ecobosque`.")
        return

    for attachment in ctx.message.attachments:
        if attachment.content_type and attachment.content_type.startswith('image/'):
            await predict_with_model(ctx, MODEL_PATHS['eco_bosque'], attachment.url)
        else:
            await ctx.send(f"Adjunta un archivo de imagen válido para usar el comando `{COMMAND_PREFIX}ecobosque`.")

@bot.command(name='ecociudad')
async def ecociudad_command(ctx):
    """Analiza una imagen con el modelo Eco_Ciudad."""
    if not ctx.message.attachments:
        await ctx.send(f"Por favor, adjunta una imagen con el comando `{COMMAND_PREFIX}ecociudad`.")
        return

    for attachment in ctx.message.attachments:
        if attachment.content_type and attachment.content_type.startswith('image/'):
            await predict_with_model(ctx, MODEL_PATHS['eco_ciudad'], attachment.url)
        else:
            await ctx.send(f"Adjunta un archivo de imagen válido para usar el comando `{COMMAND_PREFIX}ecociudad`.")

@bot.command(name='glaciares')
async def glaciares_command(ctx):
    """Analiza una imagen con el modelo Glaciares."""
    if not ctx.message.attachments:
        await ctx.send(f"Por favor, adjunta una imagen con el comando `{COMMAND_PREFIX}glaciares`.")
        return

    for attachment in ctx.message.attachments:
        if attachment.content_type and attachment.content_type.startswith('image/'):
            await predict_with_model(ctx, MODEL_PATHS['glaciares'], attachment.url)
        else:
            await ctx.send(f"Adjunta un archivo de imagen válido para usar el comando `{COMMAND_PREFIX}glaciares`.")


@bot.command(name='detalles')
async def show_details_and_advice(ctx):
    """Muestra los porcentajes de la última imagen analizada y consejos de mejora."""
    if ctx.author.id not in bot.last_prediction_data:
        await ctx.send("No hay una predicción reciente para mostrar detalles. Primero, usa uno de los comandos de análisis de imagen (ej. `!ecobosque` con una imagen).")
        return
    
    data = bot.last_prediction_data[ctx.author.id]
    
    if data['channel_id'] != ctx.channel.id:
        await ctx.send("Este comando muestra detalles de tu última predicción, pero parece que estás en un canal diferente. Por favor, usa `!detalles` en el mismo canal donde hiciste la predicción.")
        return

    await _send_details_and_advice(ctx, data['model_name_key'], data['all_probabilities'], data['good_class_name'])
    

    del bot.last_prediction_data[ctx.author.id]


bot.run(DISCORD_BOT_TOKEN)
