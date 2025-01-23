module Comparison exposing (..)

import Array
import Browser
import Browser.Navigation
import Dict
import File
import File.Select
import Gradio
import Html
import Html.Attributes
import Html.Events
import Json.Decode as D
import Json.Encode as E
import Requests
import Set
import Task
import Url


main =
    Browser.application
        { init = init
        , view = view
        , update = update
        , subscriptions = \model -> Sub.none
        , onUrlRequest = onUrlRequest
        , onUrlChange = onUrlChange
        }



-- MESSAGE


type Msg
    = NoOp
    | GotImage Requests.Id (Result Gradio.Error Example)
    | GotSaeActivations Requests.Id (Result Gradio.Error (Dict.Dict VitKey (List SaeActivation)))
    | SelectExample String
    | FocusLatent String Int
    | BlurLatent String Int
      -- See https://elm-lang.org/examples/image-previews for examples of how to implement this.
    | ImageUploader ImageUploaderMsg
      -- For latent picker
    | LatentPicker LatentPickerMsg


type ImageUploaderMsg
    = Upload
    | DragEnter
    | DragLeave
    | GotFile File.File
    | GotPreview String


type LatentPickerMsg
    = Pick
    | SelectVit VitKey
    | ToggleDropdown
    | UpdateSearch String
    | SelectLatent Int



-- MODEL


type alias Model =
    { -- Browser
      key : Browser.Navigation.Key

    -- State
    , inputExample : Requests.Requested Example
    , imageUploaderHover : Bool

    -- Mapping from ViT name (string) to a list of activations. Changes as the input image changes.
    , saeActivations : Requests.Requested (Dict.Dict VitKey (List SaeActivation))
    , focusedLatent : Maybe ( VitKey, Int )

    -- Latent picker
    , latents : Dict.Dict VitKey (Set.Set Int)
    , latentPickerSearch : String
    , latentPickerLatent : Maybe Int
    , latentPickerOpen : Bool
    , latentPickerVitKey : String

    -- APIs
    , gradio : Gradio.Config
    , imageRequestId : Requests.Id
    , saeActivationsRequestId : Requests.Id
    }


type alias VitKey =
    String


type alias SaeActivation =
    { vit : VitKey
    , latent : Int
    , activations : List Float -- Has nPatches floats

    -- TODO: add examples
    , examples : List HighlightedExample
    }


type alias Example =
    { url : Gradio.Base64Image
    , label : String
    }


type alias HighlightedExample =
    { original : Gradio.Base64Image
    , highlighted : Gradio.Base64Image
    , label : String
    , exampleId : String
    }


init : () -> Url.Url -> Browser.Navigation.Key -> ( Model, Cmd Msg )
init _ url key =
    let
        model =
            { -- Browser
              key = key
            , inputExample = Requests.Initial
            , saeActivations = Requests.Initial
            , focusedLatent = Nothing
            , imageUploaderHover = False
            , latents = Dict.empty
            , latentPickerSearch = ""
            , latentPickerLatent = Nothing
            , latentPickerOpen = False
            , latentPickerVitKey = ""

            -- APIs
            , gradio =
                { host = "http://127.0.0.1:7860" }
            , imageRequestId = Requests.init
            , saeActivationsRequestId = Requests.init
            }
    in
    ( model, getImage model.gradio model.imageRequestId "inat21__train_mini__93571" )



-- UPDATE


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        NoOp ->
            ( model, Cmd.none )

        GotImage id result ->
            if Requests.isStale id model.imageRequestId then
                ( model, Cmd.none )

            else
                case result of
                    Ok example ->
                        let
                            saeActivationsRequestId =
                                Requests.next model.saeActivationsRequestId
                        in
                        ( { model
                            | inputExample = Requests.Loaded example
                            , saeActivationsRequestId = saeActivationsRequestId
                            , saeActivations = Requests.Loading
                          }
                        , getSaeActivations model.gradio
                            saeActivationsRequestId
                            model.latents
                            example.url
                        )

                    Err err ->
                        ( { model
                            | inputExample = Requests.Failed (explainGradioError err)
                          }
                        , Cmd.none
                        )

        GotSaeActivations id result ->
            if Requests.isStale id model.saeActivationsRequestId then
                ( model, Cmd.none )

            else
                case result of
                    Ok saeActivations ->
                        ( { model
                            | saeActivations = Requests.Loaded saeActivations
                          }
                        , Cmd.none
                        )

                    Err err ->
                        ( { model
                            | saeActivations = Requests.Failed (explainGradioError err)
                          }
                        , Cmd.none
                        )

        FocusLatent name latent ->
            ( { model | focusedLatent = Just ( name, latent ) }, Cmd.none )

        BlurLatent name latent ->
            case model.focusedLatent of
                Nothing ->
                    ( model, Cmd.none )

                Just ( n, l ) ->
                    if n == name && l == latent then
                        ( { model | focusedLatent = Nothing }, Cmd.none )

                    else
                        ( model, Cmd.none )

        SelectExample exampleId ->
            let
                imageRequestId =
                    Requests.next model.imageRequestId
            in
            ( { model
                | inputExample = Requests.Loading
                , imageRequestId = imageRequestId
                , saeActivations = Requests.Initial
                , focusedLatent = Nothing
              }
            , getImage model.gradio
                imageRequestId
                exampleId
            )

        ImageUploader imageMsg ->
            imageUploaderUpdate model imageMsg

        LatentPicker latentMsg ->
            latentPickerUpdate model latentMsg


imageUploaderUpdate : Model -> ImageUploaderMsg -> ( Model, Cmd Msg )
imageUploaderUpdate model msg =
    case msg of
        Upload ->
            ( model, File.Select.file [ "image/*" ] (GotFile >> ImageUploader) )

        DragEnter ->
            ( { model | imageUploaderHover = True }, Cmd.none )

        DragLeave ->
            ( { model | imageUploaderHover = False }, Cmd.none )

        GotFile file ->
            ( { model | imageUploaderHover = False }
            , Task.perform (GotPreview >> ImageUploader) <| File.toUrl file
            )

        GotPreview preview ->
            case Gradio.base64Image preview of
                Just url ->
                    let
                        saeActivationsRequestId =
                            Requests.next model.saeActivationsRequestId
                    in
                    ( { model
                        | inputExample = Requests.Loaded { url = url, label = "User Example" }
                        , saeActivationsRequestId = saeActivationsRequestId
                        , saeActivations = Requests.Loading
                      }
                    , getSaeActivations model.gradio
                        saeActivationsRequestId
                        model.latents
                        url
                    )

                Nothing ->
                    ( { model | inputExample = Requests.Failed "Uploaded image was not base64." }
                    , Cmd.none
                    )


latentPickerUpdate : Model -> LatentPickerMsg -> ( Model, Cmd Msg )
latentPickerUpdate model msg =
    case msg of
        Pick ->
            case ( model.latentPickerLatent, model.inputExample ) of
                ( Just latent, Requests.Loaded example ) ->
                    let
                        saeActivationsRequestId =
                            Requests.next model.saeActivationsRequestId

                        latents =
                            updateDictSet model.latentPickerVitKey latent model.latents
                    in
                    ( { model
                        | saeActivationsRequestId = saeActivationsRequestId
                        , saeActivations = Requests.Loading
                        , latentPickerSearch = ""
                        , latentPickerOpen = False
                        , latents = latents
                      }
                    , getSaeActivations model.gradio
                        saeActivationsRequestId
                        latents
                        example.url
                    )

                ( Nothing, _ ) ->
                    ( model, Cmd.none )

                ( _, Requests.Loading ) ->
                    ( model, Cmd.none )

                ( _, Requests.Failed _ ) ->
                    ( model, Cmd.none )

                ( _, Requests.Initial ) ->
                    ( model, Cmd.none )

        UpdateSearch str ->
            ( { model | latentPickerSearch = str, latentPickerOpen = True }
            , Cmd.none
            )

        SelectLatent number ->
            ( { model
                | latentPickerLatent = Just number
                , latentPickerOpen = False
              }
            , Cmd.none
            )

        SelectVit key ->
            ( { model | latentPickerVitKey = key }
            , Cmd.none
            )

        ToggleDropdown ->
            ( { model | latentPickerOpen = True }
            , Cmd.none
            )


onUrlRequest : Browser.UrlRequest -> Msg
onUrlRequest request =
    NoOp


onUrlChange : Url.Url -> Msg
onUrlChange url =
    NoOp



-- API


explainGradioError : Gradio.Error -> String
explainGradioError err =
    case err of
        Gradio.NetworkError msg ->
            "Network error: " ++ msg

        Gradio.JsonError msg ->
            "Error decoding JSON: " ++ msg

        Gradio.ParsingError msg ->
            "Error parsing API response: " ++ msg

        Gradio.ApiError msg ->
            "Error in the API: " ++ msg


getImage : Gradio.Config -> Requests.Id -> String -> Cmd Msg
getImage cfg id exampleId =
    Gradio.get cfg
        "get-image"
        [ E.string exampleId ]
        (D.map2 Example
            (D.index 0 Gradio.base64ImageDecoder)
            (D.index 1 D.string)
        )
        (GotImage id)


getSaeActivations : Gradio.Config -> Requests.Id -> Dict.Dict VitKey (Set.Set Int) -> Gradio.Base64Image -> Cmd Msg
getSaeActivations cfg id latents image =
    Gradio.get cfg
        "get-sae-activations"
        [ Gradio.encodeImg image, encodeLatents latents ]
        (Gradio.decodeOne
            (D.dict
                (D.list
                    (D.map4
                        SaeActivation
                        (D.field "model_cfg" (D.field "key" D.string))
                        (D.field "latent" D.int)
                        (D.field "activations" (D.list D.float))
                        (D.field "examples" (D.list highlightedExampleDecoder))
                    )
                )
            )
        )
        (GotSaeActivations id)


encodeLatents : Dict.Dict VitKey (Set.Set Int) -> E.Value
encodeLatents latents =
    E.dict identity (E.list E.int) (Dict.map (\_ set -> Set.toList set) latents)



-- Debug.todo "encodelatents"


highlightedExampleDecoder : D.Decoder HighlightedExample
highlightedExampleDecoder =
    D.map4
        HighlightedExample
        (D.field "orig_url" Gradio.base64ImageDecoder)
        (D.field "highlighted_url" Gradio.base64ImageDecoder)
        (D.field "label" D.string)
        (D.field "example_id" D.string)



-- VIEW


view : Model -> Browser.Document Msg
view model =
    { title = "Image Classification"
    , body =
        [ Html.div
            [ Html.Attributes.class "flex flex-row" ]
            [ -- Image picker, left column
              Html.div
                [ Html.Attributes.class "flex flex-col" ]
                [ viewInputExample model.focusedLatent model.saeActivations model.inputExample
                , Html.p [] [ Html.text "TODO: other example images here" ]
                , viewGenericButton (ImageUploader Upload) "Upload Image"
                , viewLatentPicker
                    model.latentPickerOpen
                    model.latentPickerSearch
                    model.latentPickerLatent
                , viewModelPicker
                , viewGenericButton (LatentPicker Pick) "Add Latent"
                ]

            -- SAE stuff, right column
            , Html.div
                [ Html.Attributes.class "flex flex-col flex-1" ]
                [ viewSaeActivations model.focusedLatent model.saeActivations ]
            ]
        ]
    }


viewInputExample : Maybe ( VitKey, Int ) -> Requests.Requested (Dict.Dict VitKey (List SaeActivation)) -> Requests.Requested Example -> Html.Html Msg
viewInputExample focusedLatent requestedSaeActivations requestedExample =
    case ( focusedLatent, requestedSaeActivations, requestedExample ) of
        ( _, _, Requests.Initial ) ->
            Html.p
                []
                [ Html.text "Loading initial example..." ]

        ( _, _, Requests.Loading ) ->
            Html.p
                []
                [ Html.text "Loading example..." ]

        ( _, _, Requests.Failed err ) ->
            viewErr err

        ( Nothing, _, Requests.Loaded example ) ->
            viewGriddedImage [] example

        ( _, Requests.Initial, Requests.Loaded example ) ->
            viewGriddedImage [] example

        ( _, Requests.Loading, Requests.Loaded example ) ->
            viewGriddedImage [] example

        ( _, Requests.Failed _, Requests.Loaded example ) ->
            viewGriddedImage [] example

        ( Just ( model, latent ), Requests.Loaded activations, Requests.Loaded example ) ->
            let
                values =
                    Dict.get model activations
                        |> Maybe.map (List.filter (\act -> act.latent == latent))
                        |> Maybe.andThen List.head
                        |> Maybe.map .activations
                        |> Maybe.withDefault []
            in
            viewGriddedImage values example


viewGriddedImage : List Float -> Example -> Html.Html Msg
viewGriddedImage values { url, label } =
    Html.div
        []
        [ Html.div
            [ Html.Attributes.class "relative inline-block" ]
            [ Html.div
                [ Html.Attributes.class "absolute grid"
                , Html.Attributes.class "grid-rows-[repeat(14,_16px)] grid-cols-[repeat(14,_16px)]"
                , Html.Attributes.class "md:grid-rows-[repeat(14,_24px)] md:grid-cols-[repeat(14,_24px)]"
                , Html.Attributes.class "lg:grid-rows-[repeat(14,_32px)] lg:grid-cols-[repeat(14,_32px)]"
                , Html.Attributes.class "xl:grid-rows-[repeat(14,_40px)] xl:grid-cols-[repeat(14,_40px)]"
                ]
                (List.map viewGridCell values)
            , Html.img
                [ Html.Attributes.class "block w-[224px] h-[224px]"
                , Html.Attributes.class "md:w-[336px] md:h-[336px]"
                , Html.Attributes.class "lg:w-[448px] lg:h-[448px]"
                , Html.Attributes.class "xl:w-[560px] xl:h-[560px]"
                , Html.Attributes.src (Gradio.base64ImageToString url)
                ]
                []
            ]
        , Html.p
            []
            [ Html.text label ]
        ]


viewGridCell : Float -> Html.Html Msg
viewGridCell value =
    let
        opacity =
            Array.fromList
                [ "opacity-0"
                , "opacity-10"
                , "opacity-20"
                , "opacity-30"
                , "opacity-40"
                , "opacity-50"
                , "opacity-60"
                , "opacity-70"
                , "opacity-80"
                , "opacity-90"
                , "opacity-100"
                ]
                |> Array.get (bucket (0.5 * value))
                |> Maybe.withDefault "opacity-0"
    in
    Html.div
        [ Html.Attributes.class "w-[16px] h-[16px] bg-rose-600"
        , Html.Attributes.class "md:w-[24px] md:h-[24px]"
        , Html.Attributes.class "lg:w-[32px] lg:h-[32px]"
        , Html.Attributes.class "xl:w-[40px] xl:h-[40px]"
        , Html.Attributes.class opacity
        ]
        []


bucket : Float -> Int
bucket value =
    -- Clamp to ensure value is in [0.0, 1.0]
    let
        clamped =
            clamp 0.0 1.0 value

        -- Multiply by 10 and floor
        -- Special case: if value is 1.0, we want bucket 9 not 10
        out =
            if clamped >= 1.0 then
                9

            else
                floor (clamped * 10)
    in
    out


viewSaeActivations : Maybe ( VitKey, Int ) -> Requests.Requested (Dict.Dict VitKey (List SaeActivation)) -> Html.Html Msg
viewSaeActivations focusedLatent requestedActivations =
    case requestedActivations of
        Requests.Initial ->
            Html.p
                [ Html.Attributes.class "italic" ]
                [ Html.text "Load an image to see SAE activations." ]

        Requests.Loading ->
            Html.p
                []
                [ Html.text "Loading SAE activations..." ]

        Requests.Loaded activations ->
            Html.div
                []
                (Dict.toList activations
                    |> List.map (uncurry (viewModelSaeActivations focusedLatent))
                )

        Requests.Failed err ->
            viewErr err


viewModelSaeActivations : Maybe ( VitKey, Int ) -> String -> List SaeActivation -> Html.Html Msg
viewModelSaeActivations focusedLatent model saeActivations =
    Html.div
        []
        [ Html.p
            []
            [ Html.text model ]
        , Html.div
            [ Html.Attributes.class "grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3" ]
            (List.map
                (viewSaeActivation focusedLatent)
                saeActivations
            )
        ]


viewSaeActivation : Maybe ( VitKey, Int ) -> SaeActivation -> Html.Html Msg
viewSaeActivation focusedLatent { vit, latent, activations, examples } =
    let
        active =
            case focusedLatent of
                Nothing ->
                    False

                Just ( name, l ) ->
                    name == vit && l == latent
    in
    Html.div []
        [ Html.details
            [ Html.Attributes.class "cursor-pointer rounded-lg border border-gray-200 bg-white shadow-sm hover:shadow-md transition-shadow duration-200 dark:border-gray-700 dark:bg-gray-800"
            , Html.Attributes.attribute "open" ""
            , Html.Events.onMouseEnter (FocusLatent vit latent)
            , Html.Events.onMouseLeave (BlurLatent vit latent)
            ]
            [ Html.summary
                [ Html.Attributes.class "cursor-pointer select-none px-4 py-3 hover:bg-gray-50 text-gray-900" ]
                [ Html.span
                    []
                    [ Html.text ("Latent: " ++ String.fromInt latent) ]
                ]
            , Html.div
                [ Html.Attributes.class "p-1 text-gray-600 border-t border-gray-200 grid gap-1 grid-cols-2 md:grid-cols-4" ]
                (List.map (viewHighlightedExample active) examples)
            ]
        ]


viewHighlightedExample : Bool -> HighlightedExample -> Html.Html Msg
viewHighlightedExample active { original, highlighted, label, exampleId } =
    let
        ( classOriginal, classHighlighted ) =
            if active then
                ( "opacity-0", "opacity-100" )

            else
                ( "opacity-100", "opacity-0" )
    in
    Html.div
        [ Html.Attributes.class "relative"
        , Html.Events.onClick (SelectExample exampleId)
        ]
        [ Html.img
            [ Html.Attributes.class "transition-opacity duration-100"
            , Html.Attributes.class classOriginal
            , Html.Attributes.src (Gradio.base64ImageToString original)
            ]
            []
        , Html.img
            [ Html.Attributes.class "absolute inset-0 transition-opacity duration-100"
            , Html.Attributes.class classHighlighted
            , Html.Attributes.src (Gradio.base64ImageToString highlighted)
            ]
            []
        ]


viewErr : String -> Html.Html Msg
viewErr err =
    Html.div
        [ Html.Attributes.class "relative rounded-lg border border-red-200 bg-red-50 p-4 m-4" ]
        [ Html.button
            []
            []
        , Html.h3
            [ Html.Attributes.class "font-bold text-red-800" ]
            [ Html.text "Error" ]
        , Html.p
            [ Html.Attributes.class "text-red-700" ]
            [ Html.text err ]
        ]



-- TODO: configure 24000


viewLatentPicker : Bool -> String -> Maybe Int -> Html.Html Msg
viewLatentPicker open search maybeLatent =
    Html.div
        [ Html.Attributes.class "relative" ]
        [ Html.input
            [ Html.Attributes.type_ "text"
            , Html.Attributes.placeholder "Search numbers (1-24576)..."
            , Html.Attributes.value search
            , Html.Events.onInput (UpdateSearch >> LatentPicker)
            , Html.Events.onClick (LatentPicker ToggleDropdown)
            , Html.Attributes.class "w-100"

            -- , style "width" "100%"
            -- , style "padding" "8px"
            -- , style "border" "1px solid #ccc"
            -- , style "border-radius" "4px"
            ]
            []
        , viewDropdown open search
        , viewSelection maybeLatent
        ]


viewDropdown : Bool -> String -> Html.Html Msg
viewDropdown open search =
    if open then
        let
            filteredNumbers =
                List.filter
                    (\n -> String.contains search (String.fromInt n))
                    (List.range 1 24576)
                    |> List.take 100

            -- Only show first 100 matches for performance
        in
        Html.div
            []
            (List.map viewNumberOption filteredNumbers)

    else
        Html.text ""


viewNumberOption : Int -> Html.Html Msg
viewNumberOption number =
    Html.div
        [ Html.Events.onClick (LatentPicker (SelectLatent number))

        -- , style "padding" "8px 16px"
        -- , style "cursor" "pointer"
        -- , style "hover:background-color" "#f3f4f6"
        ]
        [ Html.text (String.fromInt number) ]


viewSelection : Maybe Int -> Html.Html Msg
viewSelection maybeLatent =
    case maybeLatent of
        Just latent ->
            Html.div
                [ Html.Attributes.class "mt-1"
                ]
                [ Html.text ("Selected: " ++ String.fromInt latent) ]

        Nothing ->
            Html.text ""


viewModelPicker : Html.Html Msg
viewModelPicker =
    Html.div
        []
        [ Html.label
            []
            [ Html.text "Choose a model:" ]
        , Html.select
            [ Html.Events.onInput (SelectVit >> LatentPicker) ]
            (List.map viewVitOption (Dict.keys vits))
        ]


viewVitOption : VitKey -> Html.Html Msg
viewVitOption key =
    Html.option [ Html.Attributes.value key ] [ Html.text (viewVitName key) ]


viewGenericButton : Msg -> String -> Html.Html Msg
viewGenericButton msg name =
    Html.button
        [ Html.Attributes.class "rounded-lg px-2 py-1 transition-colors"
        , Html.Attributes.class "border border-sky-300 hover:border-sky-400"
        , Html.Attributes.class "bg-sky-100 hover:bg-sky-200"
        , Html.Attributes.class "text-gray-700 hover:text-gray-900"
        , Html.Attributes.class "focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
        , Html.Attributes.class "active:bg-gray-300"
        , Html.Events.onClick msg
        ]
        [ Html.text name ]



-- HELPERS


uncurry : (a -> b -> c) -> ( a, b ) -> c
uncurry f ( a, b ) =
    f a b


updateDictSet : comparable -> comparable2 -> Dict.Dict comparable (Set.Set comparable2) -> Dict.Dict comparable (Set.Set comparable2)
updateDictSet key value dict =
    Dict.update key
        (\maybeSet ->
            case maybeSet of
                Just set ->
                    Just (Set.insert value set)

                Nothing ->
                    Just (Set.singleton value)
        )
        dict



-- CONSTANTS


viewVitName : VitKey -> String
viewVitName key =
    case Dict.get key vits of
        Just name ->
            name

        Nothing ->
            "Unknown ViT key: " ++ key


vits : Dict.Dict VitKey String
vits =
    Dict.fromList
        [ ( "clip/imagenet", "CLIP & ImageNet-1K" )
        , ( "dinov2/imagenet", "DINOv2 & ImageNet-1K" )
        , ( "bioclip/inat21", "BioCLIP & iNat21" )
        , ( "clip/inat21", "CLIP & iNat21" )
        ]
